from __future__ import annotations

import base64
import copy
import gc
import io
import json
import hashlib
import os
import resource
import time
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter, resample_poly
try:
    import pyloudnorm as pyln
except Exception:  # optional dependency during local smoke tests
    pyln = None
try:
    import soundfile as sf
except Exception:  # optional dependency during local smoke tests
    sf = None

from .analyzer import analyze_audio, analyze_audio_fast_qc
from .dynamics import (
    apply_gain_db,
    dynamic_eq,
    lookahead_limiter,
    multiband_compress,
    multiband_compress_custom,
    soft_clip,
    subtle_saturation,
    transient_micro_expander,
    upward_parallel_density,
)
from .filters import apply_eq_moves, highpass
from .ms import adjust_width, ms_low_mono, lr_to_ms, ms_to_lr
from .oversampling import (
    normalize_to_true_peak_chunked,
    oversampled_limit_chunked,
    true_peak_db_oversampled_chunked,
)
from .spatial_fx import apply_spatial_fx_mastering_plan
from .temporal_drift import stabilize_temporal_spectral_drift
from .residue_centrifuge import residue_indirect_risk_summary
from .post_master_qc import (
    build_final_master_confidence,
    build_playback_translation_auditor,
    build_post_master_qc_ledger,
)
from .ai_residue_architecture import build_ai_residue_architecture_report
from .virtual_hot_solver import build_virtual_hot_commercial_solver
from .research_blocker_planner import build_research_driven_blocker_plan
from .loudness_candidate_ranker import (
    build_headroom_blocker_scores,
    build_one_shot_candidate_plan,
    score_loudness_candidate,
)
from .perceptual_damage import (
    build_lra_recovery_feasibility,
    build_microdynamic_tolerance_report,
    build_near_miss_eligibility,
    build_pde_score,
    lra_recovery_acceptance,
)
from .stem_sum_residue_witness import build_ssrw_pde_context, compact_ssrw_summary
from .section_planner import (
    apply_section_aware_gain_envelope,
    apply_section_planner_to_candidate_plans,
    build_section_aware_loudness_planner,
    section_planner_preview_score_adjustment,
)
from .pipeline_architecture import build_pipeline_architecture_state, render_budget_policy
from .adaptive_preview import build_adaptive_audio_preview_plan
from .mix_conditioning import (
    apply_mix_conditioning_audio,
    calculate_perceptual_quality_score,
)
from .virtual_proxy_render import build_virtual_proxy_candidate_plan
from .virtual_mix_conditioning_render import expand_candidate_plans_with_virtual_mix_conditioning_feedback
from .drum_transient_punch import apply_drum_transient_punch_complete, ENGINE_VERSION as V6361_DTP_ENGINE_VERSION
from .commercial_density import apply_commercial_maximum_density_push, ENGINE_VERSION as V6404_COMMERCIAL_DENSITY_ENGINE_VERSION

WORKING_STAGE_LUFS = -18.0
FINAL_TRUE_PEAK_CEILING_DB = float(os.environ.get("BUSY_TRUE_PEAK_CEILING_DB", "-1.0"))
COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB = float(os.environ.get("BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB", "-1.0"))


def _refresh_runtime_mastering_policy(decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Refresh process-level intensity and TP globals from the locked auto policy.

    The split worker imports this module before Stage A-1 writes the auto-intensity
    policy, so import-time constants can otherwise stay at the default Balanced
    ceiling.  v8.5.3.33 makes the A-1 auto_intensity_policy.final_mode the single
    source of truth inside every render/limiter worker.
    """
    global FINAL_TRUE_PEAK_CEILING_DB, COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB

    policy = {}
    if isinstance(decision, dict) and isinstance(decision.get("auto_intensity_policy"), dict):
        policy = decision.get("auto_intensity_policy") or {}

    final = str((policy.get("final_mode") if policy else None) or (decision.get("mastering_intensity") if isinstance(decision, dict) else None) or os.environ.get("BUSY_MASTERING_INTENSITY") or "balanced").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"commercial_reference": "commercial", "commercial_reference_match": "commercial", "max": "maximum", "loudest": "maximum", "hot": "maximum", "standard": "balanced", "streaming": "safe", "streaming_safe": "safe", "automatic": "auto"}
    final = aliases.get(final, final)
    if final == "auto" or final not in {"safe", "balanced", "commercial", "maximum"}:
        final = "balanced"

    default_tp = {"safe": -2.0, "balanced": -1.0, "commercial": -0.1, "maximum": -0.1}.get(final, -1.0)
    default_push = {"safe": 2.0, "balanced": 4.0, "commercial": 6.0, "maximum": 8.0}.get(final, 4.0)
    default_chase = {"safe": 0.35, "balanced": 0.60, "commercial": 0.85, "maximum": 1.0}.get(final, 0.60)

    def _float_or(value: Any, fallback: float) -> float:
        try:
            value_f = float(value)
            if np.isfinite(value_f):
                return float(value_f)
        except Exception:
            pass
        return float(fallback)

    tp = _float_or(policy.get("selected_true_peak_ceiling_dbtp") if policy else os.environ.get("BUSY_TRUE_PEAK_CEILING_DB"), default_tp)
    if final in {"commercial", "maximum"} and str(os.environ.get("BUSY_V6404_COMMERCIAL_TP_OWNERSHIP", "1")).strip().lower() not in {"0", "false", "no", "off"}:
        # v64.0.4: Commercial/Maximum ownership means the WAV master is allowed
        # to occupy the -0.1 dBTP ceiling. Codec-safe distribution is handled as
        # a separate candidate, not by silently lowering the WAV ceiling.
        tp = _float_or(os.environ.get("BUSY_V6404_TARGET_TRUE_PEAK_DBTP"), -0.1)
        tp = float(max(-0.30, min(-0.05, tp)))
    max_push = _float_or(policy.get("selected_max_boost_db") if policy else os.environ.get("BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB"), default_push)
    chase = _float_or(policy.get("selected_chase_strength") if policy else os.environ.get("BUSY_AUTO_INTENSITY_CHASE_STRENGTH"), default_chase)

    os.environ["BUSY_MASTERING_INTENSITY"] = final
    os.environ["BUSY_TRUE_PEAK_CEILING_DB"] = str(round(tp, 3))
    os.environ["BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB"] = str(round(tp, 3))
    os.environ["BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB"] = str(round(max_push, 3))
    os.environ["BUSY_COMMERCIAL_DB_TARGET_MAX_PUSH_DB"] = str(round(max_push, 3))
    os.environ["BUSY_AUTO_INTENSITY_CHASE_STRENGTH"] = str(round(chase, 3))

    FINAL_TRUE_PEAK_CEILING_DB = float(tp)
    COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB = float(tp)

    if isinstance(decision, dict):
        decision["mastering_intensity"] = final
        if policy:
            decision["mastering_intensity_locked"] = True

    return {
        "schema_version": "busy_runtime_intensity_lock_v8_5_3_33",
        "active": True,
        "final_mode": final,
        "true_peak_ceiling_dbtp": round(float(tp), 3),
        "commercial_reference_true_peak_ceiling_dbtp": round(float(COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB), 3),
        "max_push_db": round(float(max_push), 3),
        "chase_strength": round(float(chase), 3),
        "has_auto_policy": bool(policy),
        "policy": "auto_intensity_policy.final_mode is the single source of truth for downstream commercial finish, tolerance matrix and final limiter",
    }


def _rss_mb() -> float:
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return -1.0


def _debug(step: str, **data: Any) -> None:
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "step": step,
        "rss_mb": _rss_mb(),
        "build_id": os.environ.get("BUSY_BUILD_ID", "unknown"),
        **data,
    }
    line = "[BUSY_JOB] " + json.dumps(payload, ensure_ascii=False, default=str)
    print(line, flush=True)
    log_path = os.environ.get("BUSY_JOB_LOG")
    if log_path:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


def _coerce_audio_matrix_for_meter(y: np.ndarray) -> np.ndarray:
    """Return audio as (samples, channels) for measurement-only paths."""
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    elif arr.ndim == 2 and arr.shape[0] <= 8 and arr.shape[1] > max(16, arr.shape[0] * 16):
        arr = arr.T
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    return np.ascontiguousarray(arr)


def _mono_downmix_for_meter(y: np.ndarray) -> np.ndarray:
    arr = _coerce_audio_matrix_for_meter(y)
    if arr.shape[1] == 1:
        return arr[:, 0]
    return np.mean(arr[:, :2], axis=1)


def _measure_lufs_diagnostics(y: np.ndarray, sr: int) -> dict[str, Any]:
    """v62.8.1 stereo authority loudness measurement.

    Integrated LUFS for a stereo master must be measured on the delivered stereo
    channel layout.  The mono downmix remains useful as a diagnostic, but it must
    not become final_lufs/target_met authority.
    """
    arr = _coerce_audio_matrix_for_meter(y)
    mono = _mono_downmix_for_meter(arr)
    stereo_lufs = None
    mono_lufs = None
    meter_source = "pyloudnorm" if pyln is not None else "rms_fallback"
    if pyln is not None:
        try:
            meter = pyln.Meter(sr)
            stereo_lufs = float(meter.integrated_loudness(arr if arr.shape[1] > 1 else mono))
        except Exception:
            stereo_lufs = None
        try:
            meter = pyln.Meter(sr)
            mono_lufs = float(meter.integrated_loudness(mono))
        except Exception:
            mono_lufs = None
    if stereo_lufs is None:
        rms = float(np.sqrt(np.mean(np.square(arr)))) if arr.size else 0.0
        stereo_lufs = 20 * np.log10(max(rms, 1e-12)) - 3.0
    if mono_lufs is None:
        rms_m = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
        mono_lufs = 20 * np.log10(max(rms_m, 1e-12)) - 3.0
    delta = float(stereo_lufs - mono_lufs) if stereo_lufs is not None and mono_lufs is not None else None
    return {
        "schema_version": "busy_measurement_authority_v8_5_3_62_8_1",
        "authority": "stereo_integrated_lufs" if arr.shape[1] > 1 else "mono_integrated_lufs",
        "meter_source": meter_source,
        "input_shape": [int(x) for x in arr.shape],
        "stereo_integrated_lufs": round(float(stereo_lufs), 3) if stereo_lufs is not None and np.isfinite(stereo_lufs) else None,
        "mono_downmix_lufs": round(float(mono_lufs), 3) if mono_lufs is not None and np.isfinite(mono_lufs) else None,
        "stereo_minus_mono_lufs_db": round(float(delta), 3) if delta is not None and np.isfinite(delta) else None,
        "warning": "mono_downmix_lufs_differs_from_stereo_authority" if delta is not None and abs(delta) >= _env_float("BUSY_MEASUREMENT_MONO_STEREO_LUFS_WARN_DB", 1.5, 0.0, 12.0) else None,
        "policy": "Integrated LUFS authority uses delivered stereo channel layout; mono downmix LUFS is diagnostic only.",
    }


def _measure_lufs(y: np.ndarray, sr: int) -> float | None:
    diag = _measure_lufs_diagnostics(y, sr)
    val = diag.get("stereo_integrated_lufs") if diag.get("authority") == "stereo_integrated_lufs" else diag.get("mono_downmix_lufs")
    try:
        return float(val) if val is not None and np.isfinite(float(val)) else None
    except Exception:
        return None


def _runtime_safe_lra_summary(y: np.ndarray, sr: int) -> dict[str, Any]:
    """Bounded stereo short-term LRA estimate for runtime-safe final analysis.

    v62.9.3 keeps final analysis fast without letting LRA become null.  It
    measures a capped set of stereo 3 s windows, using the same stereo authority
    as integrated LUFS.  This is intentionally a final-QC estimate, not a full
    replacement for analyze_audio(...), and BUSY_FINAL_ANALYSIS_FULL=1 still
    enables the full analyzer.
    """
    out: dict[str, Any] = {
        "schema_version": "busy_runtime_safe_short_term_lra_v8_5_3_62_9_3",
        "measurement_mode": "stereo_window_integrated_lufs",
        "count": 0,
        "lra_lu": None,
        "policy": "bounded final-QC stereo short-term loudness estimate; prevents null LRA from weakening downstream damage gates",
    }
    try:
        arr = _coerce_audio_matrix_for_meter(y)
        n = int(arr.shape[0])
        if n <= 0 or int(sr) <= 0:
            out["reason"] = "empty_audio"
            return out
        win_sec = _env_float("BUSY_FINAL_ANALYSIS_LRA_WINDOW_SEC", 3.0, 1.0, 6.0)
        hop_sec = _env_float("BUSY_FINAL_ANALYSIS_LRA_HOP_SEC", 1.0, 0.25, 3.0)
        max_windows = int(_env_float("BUSY_FINAL_ANALYSIS_LRA_MAX_WINDOWS", 48.0, 4.0, 160.0))
        win = max(1, int(float(win_sec) * int(sr)))
        hop = max(1, int(float(hop_sec) * int(sr)))
        if n < win:
            val = _measure_lufs(arr, sr)
            if val is not None and np.isfinite(float(val)):
                f = round(float(val), 3)
                out.update({"median": f, "p10": f, "p95": f, "max": f, "lra_lu": 0.0, "count": 1, "short_audio": True})
            else:
                out["reason"] = "lufs_unavailable"
            return out
        starts = list(range(0, max(1, n - win + 1), hop))
        if len(starts) > max_windows:
            idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
            starts = [starts[int(i)] for i in idx]
        vals: list[float] = []
        for st in starts:
            seg = arr[int(st):int(st)+win]
            val = _measure_lufs(seg, sr)
            if val is not None and np.isfinite(float(val)):
                vals.append(float(val))
        if not vals:
            out["reason"] = "no_valid_windows"
            return out
        a = np.asarray(vals, dtype=np.float64)
        p10 = float(np.percentile(a, 10))
        p95 = float(np.percentile(a, 95))
        out.update({
            "median": round(float(np.median(a)), 3),
            "p10": round(p10, 3),
            "p95": round(p95, 3),
            "max": round(float(np.max(a)), 3),
            "lra_lu": round(max(0.0, p95 - p10), 3),
            "count": int(len(vals)),
            "window_sec": round(float(win_sec), 3),
            "hop_sec": round(float(hop_sec), 3),
            "max_windows": int(max_windows),
        })
        return out
    except Exception as exc:
        out.update({"reason": "exception", "error": str(exc)[:240]})
        return out


def _runtime_safe_final_analysis(y: np.ndarray, sr: int, cached: dict[str, Any] | None = None) -> dict[str, Any]:
    """v8.5.3.62.8.2 bounded final analyzer.

    Full analyze_audio() is still available with BUSY_FINAL_ANALYSIS_FULL=1, but
    production runtime should not block after the master buffer is already
    rendered.  This path recomputes true peak and authoritative stereo LUFS, then
    reuses fast-QC/available metadata for non-critical fields.
    """
    if _env_bool("BUSY_FINAL_ANALYSIS_FULL", False):
        return analyze_audio(y, sr)
    base: dict[str, Any] = {}
    try:
        if isinstance(cached, dict) and cached:
            base = copy.deepcopy(cached)
        else:
            base = analyze_audio_fast_qc(y, sr, true_peak_oversample=_qc_oversample_factor())
    except Exception:
        base = {}
    try:
        fast = analyze_audio_fast_qc(y, sr, true_peak_oversample=_qc_oversample_factor())
    except Exception:
        fast = base if isinstance(base, dict) else {}
    if not isinstance(base, dict) or not base:
        base = copy.deepcopy(fast) if isinstance(fast, dict) else {}
    try:
        lufs_diag = _measure_lufs_diagnostics(y, sr)
    except Exception as exc:
        lufs_diag = {"schema_version": "busy_measurement_authority_v8_5_3_62_8_2", "authority": "fast_fallback", "error": str(exc)[:240]}
    if "loudness" not in base or not isinstance(base.get("loudness"), dict):
        base["loudness"] = {}
    if isinstance(fast, dict) and isinstance(fast.get("loudness"), dict):
        # Preserve recomputed TP/peak/RMS/crest after dither/export buffer changes.
        for k in ("sample_peak_dbfs", "approx_true_peak_dbfs", "true_peak_dbfs", "rms_dbfs", "crest_factor_db"):
            if k in fast["loudness"]:
                base["loudness"][k] = fast["loudness"].get(k)
        for k in ("duration_sec", "sample_rate_hz", "channels", "band_energy_db", "ms", "quality_indices"):
            if k in fast:
                base[k] = fast.get(k)
    authority_lufs = lufs_diag.get("stereo_integrated_lufs") if lufs_diag.get("authority") == "stereo_integrated_lufs" else lufs_diag.get("mono_downmix_lufs")
    try:
        authority_lufs = float(authority_lufs) if authority_lufs is not None and np.isfinite(float(authority_lufs)) else None
    except Exception:
        authority_lufs = None
    if authority_lufs is not None:
        base["loudness"]["integrated_lufs"] = round(float(authority_lufs), 3)
    try:
        lra_summary = _runtime_safe_lra_summary(y, sr)
    except Exception:
        lra_summary = {}
    if isinstance(lra_summary, dict) and lra_summary.get("lra_lu") is not None:
        base["loudness"]["short_term_lufs"] = lra_summary
    elif not isinstance(base["loudness"].get("short_term_lufs"), dict):
        base["loudness"]["short_term_lufs"] = lra_summary if isinstance(lra_summary, dict) else {}
    base["loudness"]["stereo_integrated_lufs"] = lufs_diag.get("stereo_integrated_lufs")
    base["loudness"]["mono_downmix_lufs"] = lufs_diag.get("mono_downmix_lufs")
    base["loudness"]["lufs_authority"] = lufs_diag.get("authority")
    base["loudness"]["measurement_authority"] = lufs_diag
    base["schema_version"] = "audio_analysis_v7_measurement_authority_runtime_safe_final_v8_5_3_62_9_3"
    base["final_analysis_runtime_safe"] = True
    base["measurement_authority"] = lufs_diag
    base["integrated_lufs"] = base["loudness"].get("integrated_lufs")
    base["stereo_integrated_lufs"] = base["loudness"].get("stereo_integrated_lufs")
    base["mono_downmix_lufs"] = base["loudness"].get("mono_downmix_lufs")
    base["lufs_authority"] = base["loudness"].get("lufs_authority")
    base["approx_true_peak_dbfs"] = base["loudness"].get("approx_true_peak_dbfs")
    base["crest_factor_db"] = base["loudness"].get("crest_factor_db")
    try:
        base["lra_lu"] = base.get("loudness", {}).get("short_term_lufs", {}).get("lra_lu")
    except Exception:
        base["lra_lu"] = None
    try:
        base["correlation"] = base.get("ms", {}).get("correlation_avg")
    except Exception:
        pass
    return base


def _mode_defaults(mode: str) -> dict[str, Any]:
    m = mode.lower().replace(" ", "_")
    if "hot" in m:
        return {"target_lufs": -7.0, "ceiling_db": FINAL_TRUE_PEAK_CEILING_DB, "mb_mode": "hot", "mb_amount": 1.18, "clip_drive_db": 1.2, "clip_mix": 0.34, "allow_bold": True}
    if "streaming" in m or "safe" in m:
        return {"target_lufs": -10.5, "ceiling_db": FINAL_TRUE_PEAK_CEILING_DB, "mb_mode": "safe", "mb_amount": 0.60, "clip_drive_db": 0.0, "clip_mix": 0.0, "allow_bold": False}
    if "clean" in m:
        return {"target_lufs": -9.2, "ceiling_db": FINAL_TRUE_PEAK_CEILING_DB, "mb_mode": "auto", "mb_amount": 0.70, "clip_drive_db": 0.0, "clip_mix": 0.0, "allow_bold": False}
    return {"target_lufs": -8.4, "ceiling_db": FINAL_TRUE_PEAK_CEILING_DB, "mb_mode": "auto", "mb_amount": 0.90, "clip_drive_db": 0.35, "clip_mix": 0.12, "allow_bold": True}


def _target_from_decision(decision: dict[str, Any], mode: str) -> tuple[float, float]:
    defaults = _mode_defaults(mode)
    limiter = decision.get("limiter", {}) if isinstance(decision, dict) else {}
    targets = decision.get("targets", {}) if isinstance(decision, dict) else {}
    rng = targets.get("integrated_lufs_range") or limiter.get("integrated_lufs_range")
    if isinstance(rng, list) and len(rng) == 2:
        # The report's convention is [louder negative, quieter negative], average is stable.
        target = float(sum(rng) / 2.0)
    else:
        target = limiter.get("target_lufs") or decision.get("target_lufs") or defaults["target_lufs"]
    ceiling = FINAL_TRUE_PEAK_CEILING_DB
    if str(os.environ.get("BUSY_RESPECT_PROFILE_TP", "0")).lower() in {"1", "true", "yes"}:
        try:
            ceiling = float(targets.get("true_peak_dbtp", ceiling))
        except Exception:
            ceiling = FINAL_TRUE_PEAK_CEILING_DB
    # Mode is allowed to cap or push the profile target.
    if "Streaming" in mode or "Safe" in mode:
        target = min(float(target), -10.0)
    elif "Hot" in mode:
        target = min(float(target) + 0.5, -6.6)
    elif "Clean" in mode:
        target = min(float(target), -8.8)
    bamix_policy = _bamix_handoff_policy_from_decision(decision)
    if bamix_policy.get("active") and _env_bool("BUSY_AUTOMIX_TARGET_FROM_HANDOFF_POLICY", True):
        mode_bamix = _bamix_v6319_handoff_mode(decision)
        bamix_target = _bamix_policy_float(bamix_policy, "effective_final_target_lufs", "BUSY_AUTOMIX_MASTERING_EFFECTIVE_TARGET_LUFS", -11.5, -14.0, -7.8)
        if mode_bamix in {"commercial_density_completion", "guarded_completion"}:
            # v63.1.9: density-completion handoff is governance, not a hard quiet cap.
            # The policy-derived effective target is the authority; DB/default
            # targets may not silently push hotter unless explicitly configured.
            target = float(bamix_target)
            if _env_bool("BUSY_BAMIX_V6319_ALLOW_DB_HOTTER_THAN_POLICY", False):
                target = max(float(target), float(decision.get("target_lufs", target)) if isinstance(decision, dict) and decision.get("target_lufs") is not None else float(target))
            hottest = _env_float("BUSY_BAMIX_V6319_HOTTEST_EFFECTIVE_TARGET_LUFS", -8.8, -10.5, -7.8)
            if float(target) > float(hottest):
                target = float(hottest)
        else:
            # Transparent/fragile BAMix handoff still prefers the quieter polish target.
            target = min(float(target), float(bamix_target))
        try:
            bamix_ceiling = float(bamix_policy.get("true_peak_ceiling_dbtp"))
            if np.isfinite(bamix_ceiling):
                ceiling = min(float(ceiling), bamix_ceiling)
        except Exception:
            pass
    return float(target), float(ceiling)


def _stage_to_working_lufs(y: np.ndarray, sr: int, working_lufs: float = WORKING_STAGE_LUFS) -> tuple[np.ndarray, float]:
    current = _measure_lufs(y, sr)
    if current is None or not np.isfinite(current):
        return y.copy(), 0.0
    delta = working_lufs - current
    # Keep staging practical; the limiter stage will set final loudness.
    delta = float(np.clip(delta, -12.0, 12.0))
    return apply_gain_db(y, delta), delta


def _loudness_iteration(
    y: np.ndarray,
    sr: int,
    target_lufs: float,
    ceiling_db: float,
    limiter: dict[str, Any] | None = None,
    max_iters: int = 7,
) -> tuple[np.ndarray, dict[str, Any]]:
    out = y.copy()
    limiter = limiter or {}
    la = limiter.get("lookahead_ms", 1.0)
    rel = limiter.get("release_ms", 130.0)
    if isinstance(la, list):
        lookahead_ms = float(sum(la) / len(la))
    else:
        lookahead_ms = float(la)
    if isinstance(rel, list):
        release_ms = float(sum(rel) / len(rel))
    else:
        release_ms = float(rel)
    gr_history: list[float] = []
    for _ in range(max_iters):
        current = _measure_lufs(out, sr)
        if current is None or not np.isfinite(current):
            break
        delta = target_lufs - current
        if abs(delta) < 0.20:
            break
        step = float(np.clip(delta, -2.0, 2.0))
        gr_history.append(step)
        out = apply_gain_db(out, step)
        out = lookahead_limiter(out, sr, ceiling_db=ceiling_db, lookahead_ms=lookahead_ms, release_ms=release_ms)
    out = lookahead_limiter(out, sr, ceiling_db=ceiling_db, lookahead_ms=lookahead_ms, release_ms=release_ms)
    return out, {"target_lufs": target_lufs, "ceiling_db": ceiling_db, "iteration_gain_steps_db": [round(x, 3) for x in gr_history], "lookahead_ms": lookahead_ms, "release_ms": release_ms}



def _oversample_factor() -> int:
    import os
    try:
        val = int(os.environ.get("BUSY_OVERSAMPLE", "8"))
    except Exception:
        val = 8
    return int(np.clip(val, 2, 32))






def _env_bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        val = float(os.environ.get(name, str(default)))
    except Exception:
        val = default
    if lo is not None:
        val = max(lo, val)
    if hi is not None:
        val = min(hi, val)
    return float(val)



def _mastering_intensity_key() -> str:
    key = str(os.environ.get("BUSY_MASTERING_INTENSITY", "balanced") or "balanced").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {"commercial_reference": "commercial", "commercial_reference_match": "commercial", "max": "maximum", "loudest": "maximum", "hot": "maximum", "standard": "balanced", "streaming": "safe", "streaming_safe": "safe", "automatic": "auto"}
    key = aliases.get(key, key)
    if key == "auto":
        return "balanced"
    return key if key in {"safe", "balanced", "commercial", "maximum"} else "balanced"


def _commercial_tolerance_matrix(family_text: str = "") -> dict[str, Any]:
    """Return auto-intensity tolerance gates for commercial/mastering QC.

    v8.5.3.31 changes the policy from a visible/auto-selected intensity to an
    analysis-selected intensity.  Genre/family only supplies the prior; the final
    mode is selected earlier from actual audio risk (vocal, artifact, crest, LRA,
    low-end/phase and predicted peak stress).  These thresholds are QC gates, not
    direct EQ/limiter instructions.
    """
    preset = _mastering_intensity_key()
    txt = str(family_text or "").lower()
    if any(k in txt for k in ["edm", "electro", "house", "club", "dance", "techno", "schranz", "garage", "phonk", "dnb", "jungle"]):
        family = "edm_club"
        crest_hard = {"safe": 3.0, "balanced": 3.5, "commercial": 4.0, "maximum": 4.5}[preset]
        lra_hard = {"safe": 3.0, "balanced": 3.5, "commercial": 4.0, "maximum": 4.0}[preset]
    elif any(k in txt for k in ["hip", "trap", "rap", "drill", "rock", "metal", "band", "guitar"]):
        family = "hiphop_rock"
        crest_hard = {"safe": 5.5, "balanced": 6.0, "commercial": 7.0, "maximum": 8.0}[preset]
        lra_hard = {"safe": 3.5, "balanced": 4.0, "commercial": 4.5, "maximum": 5.0}[preset]
    elif any(k in txt for k in ["folk", "acoustic", "classical", "jazz", "ballad", "orchestral"]):
        family = "folk_acoustic"
        crest_hard = {"safe": 9.0, "balanced": 10.0, "commercial": 11.0, "maximum": 12.0}[preset]
        lra_hard = {"safe": 5.5, "balanced": 6.0, "commercial": 7.0, "maximum": 8.0}[preset]
    else:
        family = "pop_vocal"
        crest_hard = {"safe": 7.0, "balanced": 8.0, "commercial": 9.0, "maximum": 10.0}[preset]
        lra_hard = {"safe": 4.0, "balanced": 4.5, "commercial": 5.0, "maximum": 5.5}[preset]

    # Warning gates sit slightly above hard minimums.  Safe is the fallback, so its
    # hard gates are permissive; Maximum is strict and should only survive very clean material.
    crest_warn = crest_hard + {"safe": 0.35, "balanced": 0.45, "commercial": 0.55, "maximum": 0.70}[preset]
    lra_warn = lra_hard + {"safe": 0.25, "balanced": 0.35, "commercial": 0.45, "maximum": 0.60}[preset]
    tp_ceiling = float(os.environ.get("BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB", os.environ.get("BUSY_TRUE_PEAK_CEILING_DB", "-1.0")) or -1.0)
    hard_vetoes = ["true_peak_over_ceiling", "severe_clipping", "vocal_distortion", "phase_collapse", "low_end_collapse", "codec_danger"]
    if preset in {"commercial", "maximum"}:
        hard_vetoes += ["high_suno_artifact_score", "vocal_led_high_sibilance"]
    if preset == "maximum":
        hard_vetoes += ["non_trivial_clipping", "artifact_score_over_0_25", "near_ceiling_input_peak", "low_dynamic_health"]
    return {
        "schema_version": "busy_commercial_tolerance_policy_v8_5_3_31_auto_intensity",
        "intensity_preset": preset,
        "family_bucket": family,
        "true_peak_ceiling_dbtp": round(float(tp_ceiling), 3),
        "crest_warning_min_db": round(float(crest_warn), 3),
        "crest_hard_min_db": round(float(crest_hard), 3),
        "lra_warning_min_lu": round(float(lra_warn), 3),
        "lra_hard_min_lu": round(float(lra_hard), 3),
        "soft_tradeoffs": ["crest_factor_low", "lra_narrow", "mild_transient_rounding", "limiter_gr_high", "commercial_density_increase"],
        "hard_vetoes": hard_vetoes,
        "tp_policy": "Safe=-2.0, Balanced=-1.0, Commercial=-1.0/-0.8 when clean, Maximum=-0.1 only after strict clean-entry gates",
        "policy": "auto-selected intensity scales commercial tolerance; Maximum is earned by clean analysis, and deterministic hard safety vetoes remain active",
    }

def _parse_range_mid(value: Any, default: float) -> float:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(sum(float(x) for x in value[:2]) / 2.0)
        except Exception:
            return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _experimental_hot_commercial_enabled(mode: str | None = None) -> bool:
    """Adaptive hot-commercial unlock.

    v8.5.3.14 changes the previous opt-in experiment into an automatic
    residue-aware commercial path.  In Auto Commercial / Hot / Club style modes,
    the engine is allowed to clean measured blocker residue and then test hotter
    commercial candidates.  Environment variables are now only overrides:

    * BUSY_ADAPTIVE_HOT_COMMERCIAL=0 disables this automatic unlock.
    * BUSY_EXPERIMENTAL_HOT_COMMERCIAL=1 remains accepted for backward compat.
    * Safe/Clean/Streaming modes remain excluded unless explicitly allowed.
    """
    ml = str(mode or "").lower()
    if any(k in ml for k in ["streaming", "safe", "clean"]):
        return _env_bool("BUSY_ADAPTIVE_HOT_IN_SAFE_MODES", _env_bool("BUSY_EXPERIMENTAL_HOT_IN_SAFE_MODES", False))
    if _env_bool("BUSY_EXPERIMENTAL_HOT_COMMERCIAL", False):
        return True
    if not _env_bool("BUSY_ADAPTIVE_HOT_COMMERCIAL", True):
        return False
    # The user's Auto Commercial Master is expected to chase a real commercial
    # finish, not stop at a conservative -10 LUFS whenever residue risk appears.
    if any(k in ml for k in ["auto commercial", "commercial", "hot", "club", "edm", "master"]):
        return True
    return False

def _extract_frequency_band_centrifuge_context(before: dict[str, Any] | None, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the fullest Frequency-Band Centrifuge structure available."""
    candidates: list[dict[str, Any]] = []
    for obj in [before, (decision or {}).get("runtime_safety_context") if isinstance(decision, dict) else None, decision]:
        if not isinstance(obj, dict):
            continue
        c = obj.get("frequency_band_centrifuge")
        if isinstance(c, dict) and (c.get("global_summary") or c.get("per_band_analysis") or c.get("active")):
            candidates.append(c)
    if not candidates:
        return {}
    candidates.sort(key=lambda x: len(str(x)), reverse=True)
    return candidates[0]


def _centrifuge_band(c: dict[str, Any] | None, name: str) -> dict[str, Any]:
    if not isinstance(c, dict):
        return {}
    bands = c.get("per_band_analysis", {}) if isinstance(c.get("per_band_analysis", {}), dict) else {}
    b = bands.get(name, {})
    return b if isinstance(b, dict) else {}


def _centrifuge_band_float(b: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = float(b.get(key, default) or default)
        if not np.isfinite(v):
            return float(default)
        return float(np.clip(v, 0.0, 1.0))
    except Exception:
        return float(default)


def _centrifuge_band_max_reduction(c: dict[str, Any] | None, band_name: str, default: float) -> float:
    if not isinstance(c, dict):
        return float(default)
    proposed = c.get("proposed_actions") if isinstance(c.get("proposed_actions"), list) else []
    for a in proposed:
        if isinstance(a, dict) and str(a.get("band") or "") == band_name:
            try:
                return float(np.clip(float(a.get("max_reduction_db", default) or default), 0.05, float(default)))
            except Exception:
                return float(default)
    return float(default)


def _residue_band_allowed(
    b: dict[str, Any],
    band_name: str,
    *,
    min_conf: float,
    rejected: list[dict[str, Any]],
    allow_side_blocker: bool = False,
) -> bool:
    """Safety gate for direct residue attenuation.

    A40 is an analysis map, not an audio command.  A69 may attenuate only when
    residue confidence is high and musicality/protection evidence is low.  This
    prevents vocal air, hats, cowbells, lo-fi texture or ambience from being
    removed as if they were non-musical residue.
    """
    conf_v = _centrifuge_band_float(b, "residue_confidence", 0.0)
    musicality = _centrifuge_band_float(b, "musicality_score", 0.0)
    protection = _centrifuge_band_float(b, "stem_protection_score", 0.0)
    texture = _centrifuge_band_float(b, "intentional_texture_score", 0.0)
    transient = _centrifuge_band_float(b, "transient_alignment_score", 0.0)
    max_musicality = _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_MUSICALITY", 0.52, 0.20, 0.90)
    max_protection = _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_PROTECTION", 0.58, 0.20, 0.95)
    max_texture = _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_TEXTURE", 0.66, 0.20, 0.95)
    max_transient = _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_TRANSIENT_ALIGNMENT", 0.78, 0.35, 0.98)
    conf_ok = conf_v >= float(min_conf) or (allow_side_blocker and conf_v >= max(0.25, float(min_conf) - 0.12))
    gates = {
        "confidence_ok": bool(conf_ok),
        "musicality_ok": bool(musicality <= max_musicality),
        "protection_ok": bool(protection <= max_protection),
        "intentional_texture_ok": bool(texture <= max_texture),
        "transient_alignment_ok": bool(transient <= max_transient),
    }
    ok = all(gates.values())
    if not ok:
        rejected.append({
            "band": band_name,
            "reason": "residue_pre_clean_protection_gate",
            "gates": gates,
            "residue_confidence": round(conf_v, 4),
            "musicality_score": round(musicality, 4),
            "stem_protection_score": round(protection, 4),
            "intentional_texture_score": round(texture, 4),
            "transient_alignment_score": round(transient, 4),
        })
    return bool(ok)


def _axis_ledger_from_pressure(pressure: dict[tuple[str, str, int], float]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for axis, amount in sorted((pressure or {}).items(), key=lambda kv: float(kv[1]), reverse=True):
        try:
            target, typ, bin_hz = axis
        except Exception:
            continue
        items.append({"target": target, "type": typ, "bin": bin_hz, "existing_abs_db": round(float(amount), 3)})
    return {
        "schema_version": "busy_axis_ownership_ledger_v8_5_3_62_9_8",
        "active": bool(items),
        "controlled_axis_count": len(items),
        "top_axes": items[:18],
        "policy": "Existing corrective/commercial/residue-preclean pressure reserves axis budget so v58/v59 source-conditioning does not stack duplicate cuts on the same correction axis.",
    }



def _v6299_research_num(value: Any, default: float = 0.0) -> float:
    try:
        x = float(value)
        return float(x) if np.isfinite(x) else float(default)
    except Exception:
        return float(default)


def _v6299_analysis_lra_value(analysis: dict[str, Any] | None, default: float | None = None) -> float | None:
    val = _analysis_lra_lu(analysis if isinstance(analysis, dict) else {})
    if val is not None:
        return float(val)
    return default


def _v6299_band_map(before: dict[str, Any] | None, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    c = _extract_frequency_band_centrifuge_context(before if isinstance(before, dict) else {}, decision if isinstance(decision, dict) else {})
    return c.get("per_band_analysis", {}) if isinstance(c, dict) and isinstance(c.get("per_band_analysis"), dict) else {}


def _v6299_build_research_governance(before: dict[str, Any] | None, decision: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    """v62.9.9: research-governed musicality/dynamics/translation controls.

    This bundle intentionally does not add a new render process.  It converts the
    previously supplied deep-research rules into small scale/gate values consumed
    by the existing A40/A69/A72-A84/B105-B118 flow.
    """
    if not _env_bool("BUSY_V6299_RESEARCH_GOVERNANCE", True):
        return {"schema_version": "busy_v6299_research_governed_mastering", "enabled": False, "active": False, "reason": "disabled"}
    before = before if isinstance(before, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    bands = _v6299_band_map(before, decision)
    fbc = _extract_frequency_band_centrifuge_context(before, decision)
    gs = fbc.get("global_summary", {}) if isinstance(fbc, dict) and isinstance(fbc.get("global_summary"), dict) else {}
    ms = before.get("ms", {}) if isinstance(before.get("ms"), dict) else {}
    qi = before.get("quality_indices", {}) if isinstance(before.get("quality_indices"), dict) else {}
    style_text = " ".join(str(x or "") for x in [mode, decision.get("detected_style"), decision.get("genre_id"), decision.get("selected_profile")]).lower()
    texture_tokens = ["phonk", "lofi", "lo-fi", "hardwave", "drift", "industrial", "shoegaze", "rock", "metal", "garage", "glitch", "noise", "distortion", "ambient"]
    genre_texture_bias = 0.22 if any(t in style_text for t in texture_tokens) else 0.0

    protected_bands = []
    preserve_scale_by_band: dict[str, float] = {}
    for name, b in bands.items():
        if not isinstance(b, dict):
            continue
        musicality = _v6299_research_num(b.get("musicality_score"), 0.0)
        protection = _v6299_research_num(b.get("stem_protection_score"), 0.0)
        texture = max(_v6299_research_num(b.get("intentional_texture_score"), 0.0), genre_texture_bias)
        transient = _v6299_research_num(b.get("transient_alignment_score"), 0.0)
        residue = _v6299_research_num(b.get("residue_confidence"), 0.0)
        protect_score = float(np.clip(max(musicality, protection, texture, transient * 0.72) - max(0.0, residue - 0.55) * 0.25, 0.0, 1.0))
        scale = float(np.clip(1.0 - max(0.0, protect_score - 0.48) * 0.55, 0.58, 1.0))
        preserve_scale_by_band[name] = round(scale, 4)
        if protect_score >= 0.55:
            protected_bands.append({"band": name, "protect_score": round(protect_score, 4), "attenuation_scale_cap": round(scale, 4), "reason": "musical_texture_or_stem_protection"})

    temporal = fbc.get("temporal_residue_witness", {}) if isinstance(fbc, dict) and isinstance(fbc.get("temporal_residue_witness"), dict) else {}
    temporal_high = max(_v6299_research_num(gs.get("temporal_high_residue_risk"), 0.0), _v6299_research_num(temporal.get("temporal_high_residue_risk"), 0.0))
    temporal_low = max(_v6299_research_num(gs.get("temporal_low_mid_residue_risk"), 0.0), _v6299_research_num(temporal.get("temporal_low_mid_residue_risk"), 0.0))
    section_consistency = {
        "schema_version": "busy_v6299_section_aware_spectral_consistency",
        "enabled": bool(_env_bool("BUSY_V6299_SECTION_SPECTRAL_CONSISTENCY", True)),
        "active": bool(max(temporal_high, temporal_low) >= 0.30),
        "temporal_high_residue_risk": round(float(temporal_high), 4),
        "temporal_low_mid_residue_risk": round(float(temporal_low), 4),
        "governance": "raises suspicion only when spectral drift is weakly explained by broad arrangement movement; no new render loop",
    }

    lra = _v6299_analysis_lra_value(before, default=None)
    crest = _v6299_research_num(before.get("crest_factor_db"), 10.0)
    lra_flat = float(np.clip(((1.85 - float(lra)) / 1.40), 0.0, 1.0)) if lra is not None else 0.0
    crest_flat = float(np.clip((9.2 - crest) / 3.5, 0.0, 1.0))
    micro_scale = float(np.clip(1.0 - 0.30 * lra_flat - 0.14 * crest_flat, 0.58, 1.0))
    upward_scale = float(np.clip(1.0 - 0.55 * lra_flat - 0.18 * crest_flat, 0.30, 1.0))
    micro = {
        "schema_version": "busy_v6299_microdynamics_lra_shape_recovery",
        "enabled": bool(_env_bool("BUSY_V6299_LRA_SHAPE_RECOVERY", True)),
        "active": bool(max(lra_flat, crest_flat) >= 0.20),
        "input_lra_lu": round(float(lra), 3) if lra is not None else None,
        "input_crest_factor_db": round(float(crest), 3),
        "flat_lra_pressure": round(float(lra_flat), 4),
        "crest_fragility_pressure": round(float(crest_flat), 4),
        "dynamic_eq_intensity_scale": round(float(micro_scale), 4),
        "multiband_amount_scale": round(float(micro_scale), 4),
        "upward_density_scale": round(float(upward_scale), 4),
        "policy": "Protect microdynamics by scaling existing dynamics stages; does not add expansion or loudness push.",
    }

    side_low = _v6299_research_num(ms.get("side_ratio_lowband", ms.get("side_ratio_low", 0.0)), 0.0)
    lowmid_residue = _v6299_research_num(gs.get("low_mid_residue_risk"), 0.0)
    bass_translation_risk = float(np.clip(max(side_low * 1.9, lowmid_residue * 0.75, max(0.0, 8.8 - crest) / 3.6), 0.0, 1.0))
    low_end_control_scale = float(np.clip(0.86 + 0.26 * bass_translation_risk, 0.72, 1.16))
    translation_low_end = {
        "schema_version": "busy_v6299_translation_aware_low_end_governor",
        "enabled": bool(_env_bool("BUSY_V6299_TRANSLATION_LOW_END_GOVERNOR", True)),
        "active": bool(bass_translation_risk >= 0.22),
        "low_side_ratio": round(float(side_low), 4),
        "low_mid_residue_risk": round(float(lowmid_residue), 4),
        "bass_translation_risk": round(float(bass_translation_risk), 4),
        "low_end_control_scale": round(float(low_end_control_scale), 4),
        "policy": "Scale existing low-end control for playback translation; never boosts sub just to sound bigger.",
    }

    musical_texture = {
        "schema_version": "busy_v6299_musical_texture_preservation_gate",
        "enabled": bool(_env_bool("BUSY_V6299_MUSICAL_TEXTURE_PRESERVATION", True)),
        "active": bool(protected_bands),
        "genre_texture_bias": round(float(genre_texture_bias), 4),
        "protected_bands": protected_bands[:12],
        "attenuation_scale_by_band": preserve_scale_by_band,
        "policy": "Protect intentional hats, vocal air, grit, cowbell, ambience and lo-fi texture from being classified as residue.",
    }

    final_arbiter = {
        "schema_version": "busy_v6299_final_perceptual_ab_arbiter_policy",
        "enabled": bool(_env_bool("BUSY_V6299_FINAL_PERCEPTUAL_ARBITER", True)),
        "active": True,
        "policy": "Final replacement candidates must be louder enough and perceptually safer; cleaner quieter standard final can win over damaged loudness.",
    }
    active = bool(any(x.get("active") for x in [section_consistency, micro, translation_low_end, musical_texture, final_arbiter] if isinstance(x, dict)))
    return {
        "schema_version": "busy_v6299_research_governed_mastering",
        "enabled": True,
        "active": active,
        "conservative_apply": not bool(_env_bool("BUSY_V6299_AGGRESSIVE_APPLY", False)),
        "section_aware_spectral_consistency": section_consistency,
        "microdynamics_lra_shape_recovery": micro,
        "translation_aware_low_end_governor": translation_low_end,
        "musical_texture_preservation_gate": musical_texture,
        "final_perceptual_ab_arbiter_policy": final_arbiter,
        "policy": "Research-derived governance bundle; no new AI judge, no Stage M, no stem audio injection, no heavy proxy render loop.",
    }


def _attach_v6299_research_governance(before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    d = copy.deepcopy(decision or {})
    gov = _v6299_build_research_governance(before, d, mode)
    d["v62_9_9_research_governance"] = gov
    d.setdefault("final_decision_context", {})["v62_9_9_research_governance"] = gov
    return d, gov


def _v6299_candidate_quality_metrics(analysis: dict[str, Any] | None) -> dict[str, float | None]:
    analysis = analysis if isinstance(analysis, dict) else {}
    return {
        "lufs": _v6299_research_num(analysis.get("integrated_lufs"), -99.0),
        "true_peak_dbtp": _v6299_research_num(analysis.get("approx_true_peak_dbfs", analysis.get("true_peak_dbtp", -99.0)), -99.0),
        "crest_factor_db": _v6299_research_num(analysis.get("crest_factor_db"), 0.0),
        "lra_lu": _v6299_analysis_lra_value(analysis, default=None),
        "correlation": _v6299_research_num(analysis.get("correlation"), 1.0),
    }


def _v6299_final_perceptual_ab_arbiter(before: dict[str, Any] | None, standard_analysis: dict[str, Any] | None, candidate_analysis: dict[str, Any] | None, decision: dict[str, Any] | None, problem_report: dict[str, Any] | None = None) -> dict[str, Any]:
    if not _env_bool("BUSY_V6299_FINAL_PERCEPTUAL_ARBITER", True):
        return {"schema_version": "busy_v6299_final_perceptual_ab_arbiter", "enabled": False, "active": False, "accepted": True, "reason": "disabled"}
    std = _v6299_candidate_quality_metrics(standard_analysis)
    cand = _v6299_candidate_quality_metrics(candidate_analysis)
    before_lra = _v6299_analysis_lra_value(before if isinstance(before, dict) else {}, default=None)
    lufs_delta = float((cand.get("lufs") or -99.0) - (std.get("lufs") or -99.0))
    crest_delta = float((cand.get("crest_factor_db") or 0.0) - (std.get("crest_factor_db") or 0.0))
    lra_delta = None
    if cand.get("lra_lu") is not None and std.get("lra_lu") is not None:
        lra_delta = float(cand["lra_lu"] - std["lra_lu"])  # type: ignore[index]
    reasons: list[str] = []
    reject = False
    min_lufs_gain = _env_float("BUSY_V6299_ARBITER_MIN_USEFUL_LUFS_GAIN", 0.18, 0.0, 1.0)
    max_lra_loss = _env_float("BUSY_V6299_ARBITER_MAX_LRA_LOSS_LU", 0.18, 0.0, 1.0)
    max_crest_loss = _env_float("BUSY_V6299_ARBITER_MAX_CREST_LOSS_DB", 0.35, 0.0, 2.0)
    if lufs_delta < min_lufs_gain:
        reject = True; reasons.append("candidate_not_usefully_louder")
    if lra_delta is not None and lra_delta < -max_lra_loss:
        reject = True; reasons.append("candidate_lra_shape_loss")
    if crest_delta < -max_crest_loss:
        reject = True; reasons.append("candidate_crest_transient_loss")
    critical = (problem_report or {}).get("critical_new_vs_base") if isinstance(problem_report, dict) else []
    if critical:
        reject = True; reasons.append("candidate_new_critical_problem_set")
    # If input was already flat, be stricter about accepting a louder but flatter final.
    if before_lra is not None and before_lra < 2.0 and lra_delta is not None and lra_delta < -0.05 and lufs_delta < 0.45:
        reject = True; reasons.append("flat_input_louder_candidate_not_worth_lra_loss")
    score_standard = 100.0
    score_candidate = 100.0 + min(5.0, max(0.0, lufs_delta) * 2.0) + min(2.0, max(0.0, crest_delta))
    if lra_delta is not None:
        score_candidate += min(2.0, max(0.0, lra_delta)) - min(8.0, max(0.0, -lra_delta) * 12.0)
    score_candidate -= min(8.0, max(0.0, -crest_delta) * 5.0)
    if critical:
        score_candidate -= 18.0
    return {
        "schema_version": "busy_v6299_final_perceptual_ab_arbiter",
        "enabled": True,
        "active": True,
        "accepted": bool(not reject),
        "reject": bool(reject),
        "reasons": reasons,
        "lufs_delta_db": round(float(lufs_delta), 3),
        "lra_delta_lu": round(float(lra_delta), 3) if lra_delta is not None else None,
        "crest_delta_db": round(float(crest_delta), 3),
        "standard_metrics": std,
        "candidate_metrics": cand,
        "score_standard": round(float(score_standard), 3),
        "score_candidate": round(float(score_candidate), 3),
        "policy": "A louder replacement must be usefully louder and not perceptually flatter/damaged; true peak remains a ceiling, not a target.",
    }


def _v6299_normalize_commercial_finish_after_guard(commercial_finish_report: dict[str, Any] | None, standard_final_reference: dict[str, Any] | None, *, final_source: str = "standard_final_limiter_output") -> dict[str, Any]:
    """v62.9.9.1: keep commercial-finish telemetry aligned when a hot candidate is rejected.

    The v62.9.9 arbiter can correctly reject a louder commercial candidate and
    keep the standard final. In that case, public report fields such as selected,
    output, target_met, and reason must describe the delivered final rather than
    the rejected candidate, while preserving the rejected candidate under
    rejected_candidate_* for debugging.
    """
    if not isinstance(commercial_finish_report, dict):
        return {}
    if not bool(commercial_finish_report.get("candidate_rejected_after_final_guard")):
        return commercial_finish_report
    standard = standard_final_reference if isinstance(standard_final_reference, dict) else {}
    commercial_finish_report.setdefault("rejected_candidate_label", commercial_finish_report.get("selected"))
    if isinstance(commercial_finish_report.get("output"), dict):
        commercial_finish_report.setdefault("rejected_candidate_output", dict(commercial_finish_report.get("output") or {}))
    commercial_finish_report["selected"] = final_source
    commercial_finish_report["selected_final_source"] = final_source
    commercial_finish_report["fallback_final_source"] = final_source
    commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
    commercial_finish_report["delivered_final_is_standard_after_v6299_arbiter"] = True
    commercial_finish_report["target_met"] = False
    commercial_finish_report["target_reached"] = False
    commercial_finish_report["output"] = dict(standard) if isinstance(standard, dict) else {}
    commercial_finish_report["delivered_final_output"] = dict(standard) if isinstance(standard, dict) else {}
    commercial_finish_report["reason"] = "v62_9_9_final_perceptual_ab_arbiter_rejected_candidate_standard_final_retained"
    commercial_finish_report["finish_reason"] = commercial_finish_report["reason"]
    try:
        commercial_finish_report["final_lufs"] = round(float(standard.get("integrated_lufs")), 3)
    except Exception:
        pass
    try:
        commercial_finish_report["final_true_peak_dbtp"] = round(float(standard.get("true_peak_dbtp", standard.get("approx_true_peak_dbfs"))), 3)
    except Exception:
        pass
    commercial_finish_report["telemetry_consistency_v62_9_9_1"] = {
        "active": True,
        "candidate_rejected_after_final_guard": True,
        "public_selected_describes_delivered_final": True,
        "rejected_candidate_preserved": bool(commercial_finish_report.get("rejected_candidate_output")),
        "policy": "When the v62.9.9 arbiter rejects a louder candidate, commercial_loudness_finish.output is rewritten to the delivered standard final and the rejected hot candidate is preserved separately.",
    }
    return commercial_finish_report

def _experimental_residue_control_plan(before: dict[str, Any], decision: dict[str, Any], mode: str, stage: str = "pre_commercial_finish") -> dict[str, Any]:
    """Plan adaptive direct-light residue control.

    v8.5.3.14: this is no longer a hand-enabled exception path.  In Auto
    Commercial Master it reads the measured Centrifuge residue / side-width
    blockers and automatically applies small corrective controls when the
    numbers justify it.  Optional environment variables only tune/disable the
    behavior; they are not required for the normal commercial path.
    """
    if not _env_bool("BUSY_ADAPTIVE_RESIDUE_CONTROL", True):
        return {"active": False, "reason": "adaptive_residue_control_disabled", "stage": stage}
    mode_l = str(mode or "").lower()
    if any(k in mode_l for k in ["streaming", "safe", "clean"]) and not _env_bool("BUSY_ADAPTIVE_RESIDUE_IN_SAFE_MODES", False):
        return {"active": False, "reason": "safe_or_clean_mode", "stage": stage}
    if not _experimental_hot_commercial_enabled(mode) and not _env_bool("BUSY_ADAPTIVE_RESIDUE_WITHOUT_COMMERCIAL", False):
        return {"active": False, "reason": "non_commercial_mode", "stage": stage}

    c = _extract_frequency_band_centrifuge_context(before, decision)
    if not isinstance(c, dict) or not c:
        return {"active": False, "reason": "frequency_band_centrifuge_missing", "stage": stage}
    gs = c.get("global_summary", {}) if isinstance(c.get("global_summary", {}), dict) else {}
    q = before.get("quality_indices", {}) if isinstance(before.get("quality_indices", {}), dict) else {}
    ms = before.get("ms", {}) if isinstance(before.get("ms", {}), dict) else {}

    strength = _env_float("BUSY_ADAPTIVE_RESIDUE_STRENGTH", _env_float("BUSY_AI_RESIDUE_DIRECT_STRENGTH", 1.0, 0.20, 2.0), 0.20, 2.0)
    min_conf = _env_float("BUSY_ADAPTIVE_RESIDUE_MIN_CONF", _env_float("BUSY_AI_RESIDUE_DIRECT_MIN_CONF", 0.45, 0.25, 0.95), 0.25, 0.95)
    high_risk = float(gs.get("high_band_residue_risk", 0.0) or 0.0)
    side_high_risk = float(gs.get("side_high_residue_risk", 0.0) or 0.0)
    side_low = float(ms.get("side_ratio_lowband", 0.0) or 0.0)
    side_high = float(ms.get("side_ratio_8k_14k", 0.0) or 0.0)

    br = _centrifuge_band(c, "brilliance_7000_12000")
    air = _centrifuge_band(c, "air_12000_18000")
    ultra = _centrifuge_band(c, "ultra_air_18000_plus")
    pres = _centrifuge_band(c, "presence_4000_7000")
    upper = _centrifuge_band(c, "upper_mid_1500_4000")

    def conf(b: dict[str, Any]) -> float:
        try:
            return float(b.get("residue_confidence", 0.0) or 0.0)
        except Exception:
            return 0.0

    def side_ratio(b: dict[str, Any]) -> float:
        try:
            return float(b.get("side_ratio", 0.0) or 0.0)
        except Exception:
            return 0.0

    static_moves: list[dict[str, Any]] = []
    dynamic_moves: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []
    max_global_cut = _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_REDUCTION_DB", 1.20, 0.20, 2.50)

    # Low-side bloom is the safest direct cleanup and often unlocks limiter headroom.
    low_side_amount = 0.0
    if side_low >= _env_float("BUSY_AI_RESIDUE_LOW_SIDE_TRIGGER", 0.08, 0.03, 0.25):
        low_mid_band = _centrifuge_band(c, "low_mid_180_500")
        bass_protection = _centrifuge_band_float(low_mid_band, "stem_protection_score", 0.0)
        protection_scale = float(np.clip(1.0 - bass_protection * 0.42, 0.55, 1.0))
        low_side_amount = float(np.clip((0.52 + side_low * 2.15 * strength) * protection_scale, 0.34, 0.88))
        side_cut_cap = min(max_global_cut, _centrifuge_band_max_reduction(c, "low_mid_180_500", 0.70))
        static_moves.append({"type": "bell", "freq": 125, "gain_db": round(-min(0.45 * strength * protection_scale, side_cut_cap), 3), "q": 0.85, "target": "side", "source": "experimental_low_side_bloom_control"})
        dynamic_moves.append({"type": "bell", "freq": 92, "gain_db": round(-min(0.85 * strength * protection_scale, side_cut_cap), 3), "q": 0.85, "target": "side", "source": "experimental_low_side_bloom_control"})
        actions.append({"action": "low_side_bloom_control", "amount": round(low_side_amount, 3), "trigger_side_low": round(side_low, 5), "protection_scale": round(protection_scale, 4)})

    # 7-12 kHz side shimmer / brittle top.  This is where Suno fizz often becomes
    # exposed by limiting, but keep it side-weighted unless the full high band is risky.
    br_conf = conf(br)
    br_trigger = bool(br_conf >= min_conf or side_high >= 0.42 or side_high_risk >= 0.32)
    if br_trigger and _residue_band_allowed(br, "brilliance_7000_12000", min_conf=min_conf, rejected=rejected_candidates, allow_side_blocker=side_high >= 0.42 or side_high_risk >= 0.32):
        cut = float(np.clip(0.45 + max(0.0, br_conf - min_conf) * 2.4 + max(0.0, side_high - 0.42) * 1.1, 0.35, min(1.55, max_global_cut, _centrifuge_band_max_reduction(c, "brilliance_7000_12000", 1.20)))) * strength
        dynamic_moves.append({"type": "bell", "freq": 9200, "gain_db": -cut, "q": 1.45, "target": "side", "source": "experimental_side_high_fizz_control"})
        actions.append({"action": "side_high_fizz_control", "band": "brilliance_7000_12000", "attenuation_db": round(cut, 3), "confidence": round(br_conf, 4)})

    # 12-18 kHz synthetic air.  If side ratio is low, use a very gentle stereo
    # guard because the residue may be centered/high-wide rather than side-only.
    air_conf = conf(air)
    if air_conf >= min_conf and _residue_band_allowed(air, "air_12000_18000", min_conf=min_conf, rejected=rejected_candidates):
        side_pref = side_ratio(air) >= 0.08 or side_high_risk >= 0.35
        target = "side" if side_pref else "stereo"
        cut = float(np.clip(0.36 + max(0.0, air_conf - min_conf) * 2.0, 0.30, min(1.20, max_global_cut, _centrifuge_band_max_reduction(c, "air_12000_18000", 1.20)))) * strength
        dynamic_moves.append({"type": "bell", "freq": 14500, "gain_db": -cut, "q": 0.95, "target": target, "source": "experimental_synthetic_air_control"})
        static_moves.append({"type": "high_shelf", "freq": 13200, "gain_db": round(-min(0.16 * strength, max_global_cut * 0.33), 3), "q": 0.70, "target": target, "source": "experimental_synthetic_air_control"})
        actions.append({"action": "synthetic_air_control", "band": "air_12000_18000", "target": target, "attenuation_db": round(cut, 3), "confidence": round(air_conf, 4)})

    # 18k+ ultra-air hash.  Keep this subtle; it is mostly about avoiding a brittle
    # ultrasonic/near-Nyquist edge before pushing the limiter harder.
    ultra_conf = conf(ultra)
    if ultra_conf >= min_conf and _residue_band_allowed(ultra, "ultra_air_18000_plus", min_conf=min_conf, rejected=rejected_candidates):
        target = "side" if side_ratio(ultra) >= 0.06 or side_high_risk >= 0.35 else "stereo"
        cut = float(np.clip(0.28 + max(0.0, ultra_conf - min_conf) * 1.6, 0.25, min(0.95, max_global_cut, _centrifuge_band_max_reduction(c, "ultra_air_18000_plus", 0.95)))) * strength
        dynamic_moves.append({"type": "bell", "freq": 19000, "gain_db": -cut, "q": 0.80, "target": target, "source": "experimental_ultra_air_hash_control"})
        actions.append({"action": "ultra_air_hash_control", "band": "ultra_air_18000_plus", "target": target, "attenuation_db": round(cut, 3), "confidence": round(ultra_conf, 4)})

    # Presence/upper-mid edge remains conservative because it can overlap vocals.
    pres_conf = max(conf(pres), conf(upper))
    vocal_role = 0.0
    try:
        role = decision.get("final_decision_context", {}).get("role_first_dsp_contract", {}).get("role_importance", {})
        if isinstance(role, dict):
            vocal_role = float(role.get("vocal_front_focus", 0.0) or 0.0)
    except Exception:
        vocal_role = 0.0
    harsh = float(q.get("harshness_index_db", -99.0) or -99.0)
    pres_band = pres if conf(pres) >= conf(upper) else upper
    pres_band_name = "presence_4000_7000" if conf(pres) >= conf(upper) else "upper_mid_1500_4000"
    if pres_conf >= min_conf + 0.12 and harsh > -3.2 and vocal_role < 0.72 and _residue_band_allowed(pres_band, pres_band_name, min_conf=min_conf + 0.12, rejected=rejected_candidates):
        cut = float(np.clip(0.20 + max(0.0, pres_conf - min_conf) * 1.2, 0.18, min(0.75, max_global_cut, _centrifuge_band_max_reduction(c, pres_band_name, 0.75)))) * strength
        dynamic_moves.append({"type": "bell", "freq": 5200, "gain_db": -cut, "q": 1.45, "target": "stereo", "source": "experimental_presence_edge_control"})
        actions.append({"action": "presence_edge_control", "band": pres_band_name, "attenuation_db": round(cut, 3), "confidence": round(pres_conf, 4)})

    if not actions:
        return {
            "active": False,
            "reason": "no_direct_control_candidate_met_threshold",
            "stage": stage,
            "inputs": {
                "high_band_residue_risk": round(high_risk, 4),
                "side_high_residue_risk": round(side_high_risk, 4),
                "side_low": round(side_low, 5),
                "min_confidence": round(min_conf, 3),
                "rejected_candidate_count": len(rejected_candidates),
            },
            "rejected_candidates": rejected_candidates[:12],
        }

    return {
        "active": True,
        "schema_version": "busy_adaptive_direct_residue_light_control_v8_5_3_62_9_8",
        "stage": stage,
        "mode": "adaptive_commercial_residue_light_control",
        "strength": round(strength, 3),
        "min_confidence": round(min_conf, 3),
        "low_side_mono_amount": round(low_side_amount, 3),
        "static_moves": static_moves,
        "dynamic_moves": dynamic_moves,
        "actions": actions,
        "rejected_candidates": rejected_candidates[:12],
        "safety_gates": {
            "max_global_reduction_db": round(float(max_global_cut), 3),
            "max_musicality": _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_MUSICALITY", 0.52, 0.20, 0.90),
            "max_protection": _env_float("BUSY_RESIDUE_PRE_CLEAN_MAX_PROTECTION", 0.58, 0.20, 0.95),
            "policy": "A40 residue candidates must pass musicality/protection/texture/transient gates before A69 attenuation."
        },
        "inputs": {
            "high_band_residue_risk": round(high_risk, 4),
            "side_high_residue_risk": round(side_high_risk, 4),
            "side_low": round(side_low, 5),
            "side_high": round(side_high, 5),
            "brilliance_confidence": round(br_conf, 4),
            "air_confidence": round(air_conf, 4),
            "ultra_air_confidence": round(ultra_conf, 4),
            "presence_confidence": round(pres_conf, 4),
        },
        "policy": "automatic_when_centrifuge_residue_or_side_blockers_meet_threshold; applies measured light residue/bloom controls before louder commercial candidate testing; no broad denoising or watermark/provenance removal",
    }


def _apply_experimental_residue_light_control(y: np.ndarray, sr: int, before: dict[str, Any], decision: dict[str, Any], mode: str, stage: str = "pre_commercial_finish") -> tuple[np.ndarray, dict[str, Any]]:
    plan = _experimental_residue_control_plan(before, decision, mode, stage=stage)
    if not plan.get("active"):
        return y, plan
    out = y
    if float(plan.get("low_side_mono_amount", 0.0) or 0.0) > 0:
        out = ms_low_mono(out, sr, cutoff_hz=145.0, amount=float(plan.get("low_side_mono_amount", 0.0) or 0.0))
    static_moves = plan.get("static_moves", []) or []
    dynamic_moves = plan.get("dynamic_moves", []) or []
    if static_moves:
        out = apply_eq_moves(out, sr, static_moves, scale=1.0)
    if dynamic_moves:
        out = dynamic_eq(out, sr, dynamic_moves, intensity=1.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    plan = dict(plan)
    plan.update({
        "applied": True,
        "static_move_count": len(static_moves),
        "dynamic_move_count": len(dynamic_moves),
    })
    return out, plan




def _merge_fast_qc_into_analysis(base: dict[str, Any], fast: dict[str, Any], *, label: str) -> dict[str, Any]:
    """Return an analysis object that keeps expensive A-1 evidence but uses cleaned-audio QC metrics.

    Stage A-1 computes genre/role/AI evidence from the source/preflight-cleaned audio.
    The residue pre-clean pass may change level, M/S ratios and quality indices before
    mastering.  Downstream governors should see the cleaned metrics, while the original
    evidence and private/reference context remain available for protection decisions.
    """
    merged = copy.deepcopy(base or {})
    fast = fast or {}
    for key in [
        "duration_sec", "sample_rate_hz", "channels", "loudness", "band_energy_db",
        "ms", "quality_indices", "heuristic_tags", "integrated_lufs", "lra_lu",
        "approx_true_peak_dbfs", "crest_factor_db", "correlation",
    ]:
        if key in fast:
            merged[key] = fast.get(key)
    merged["analysis_lineage"] = {
        "schema_version": "busy_analysis_lineage_v8_5_3_16",
        "current_basis": label,
        "original_analysis_preserved": True,
        "policy": "genre/role evidence is preserved from A-1; loudness, M/S and QC metrics are updated after conditional pre-clean",
    }
    return merged


def _apply_residue_pre_clean_once(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any],
    decision: dict[str, Any],
    mode: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Conditional one-time pre-master cleanup for non-musical residue blockers.

    This is the structural v8.5.3.15 change: Centrifuge/AI-residue and temporal-drift
    handling happens before the premaster chain, not repeatedly after mastering.  If no
    measured blocker passes the threshold, the stage is a BYPASS and the original audio
    is used.  If cleanup is applied, downstream pre-master and commercial push use the
    cleaned audio and cleaned QC metrics.
    """
    before = before or {}
    report: dict[str, Any] = {
        "schema_version": "busy_residue_pre_clean_stage_v8_5_3_62_9_8",
        "stage": "pre_clean_before_premaster",
        "mode": "conditional_one_time_pre_master_cleanup",
        "enabled": _env_bool("BUSY_RESIDUE_PRE_CLEAN", True),
        "active": False,
        "bypassed": False,
        "actions": [],
        "policy": "Run once after basic analysis/Centrifuge/role evidence and before any premaster/commercial push. If nothing needs cleanup, bypass. No repeated post-master residue trimming.",
        "original_metrics": {
            "integrated_lufs": before.get("integrated_lufs"),
            "true_peak": before.get("approx_true_peak_dbfs"),
            "crest": before.get("crest_factor_db"),
            "correlation": before.get("correlation"),
            "side_low": ((before.get("ms") or {}) if isinstance(before.get("ms"), dict) else {}).get("side_ratio_lowband"),
            "side_high": ((before.get("ms") or {}) if isinstance(before.get("ms"), dict) else {}).get("side_ratio_8k_14k"),
        },
    }
    if not report["enabled"]:
        report.update({"bypassed": True, "reason": "disabled"})
        return y, report, before

    out = np.asarray(y, dtype=np.float32 if str(y.dtype) == "float32" else y.dtype).copy()

    # Temporal drift is a structural cleanup, not a residue band trim.  It should be
    # done once before premastering.  In the normal Cloud Run path preflight_clean_audio
    # already did it before analysis, so we only record that and do not repeat it.
    preflight = before.get("preflight_clean", {}) if isinstance(before.get("preflight_clean"), dict) else {}
    preflight_drift = preflight.get("temporal_spectral_drift", {}) if isinstance(preflight.get("temporal_spectral_drift"), dict) else {}
    drift_report: dict[str, Any]
    if preflight_drift:
        drift_report = {
            "enabled": True,
            "active": bool(preflight_drift.get("active")),
            "source": "preflight_clean_audio",
            "reason": preflight_drift.get("reason"),
            "selected_corrections_count": preflight_drift.get("selected_corrections_count", 0),
            "policy": "already evaluated once before A-1 analysis; not repeated in premaster/render stages",
        }
    elif _env_bool("BUSY_TEMPORAL_DRIFT_PRE_CLEAN_ONCE", True) and _env_bool("BUSY_TEMPORAL_DRIFT_STABILIZER", True):
        try:
            out, raw_drift = stabilize_temporal_spectral_drift(out, sr)
            drift_report = dict(raw_drift or {})
            drift_report["source"] = "residue_pre_clean_stage"
            drift_report["policy"] = "conditional one-time temporal drift stabilization before premaster"
            if drift_report.get("active"):
                report["active"] = True
                report["actions"].append("temporal_spectral_drift_stabilized")
        except Exception as exc:
            drift_report = {"enabled": True, "active": False, "reason": "error", "error_type": type(exc).__name__, "error": str(exc)[:300]}
    else:
        drift_report = {"enabled": False, "active": False, "reason": "disabled"}
    report["temporal_spectral_drift"] = drift_report

    # Centrifuge/architecture-based residue cleanup.  This is conditional: if the
    # measured residue/low-side blockers do not pass the thresholds, it is a bypass.
    residue_audio, residue_plan = _apply_experimental_residue_light_control(out, sr, before, decision, mode, stage="pre_clean_before_premaster")
    report["residue_light_control"] = residue_plan
    if residue_plan.get("active") and residue_plan.get("applied"):
        out = residue_audio
        report["active"] = True
        report["actions"].extend([a.get("action") for a in residue_plan.get("actions", []) if isinstance(a, dict) and a.get("action")])

    if not report["active"]:
        report.update({"bypassed": True, "reason": residue_plan.get("reason") or drift_report.get("reason") or "no_residue_or_drift_cleanup_needed"})
        return y, report, before

    cleaned_fast = analyze_audio_fast_qc(out, sr, true_peak_oversample=_qc_oversample_factor())
    out, makeup_report, cleaned_fast = _pre_clean_makeup_compensation(out, sr, before, cleaned_fast)
    report["pre_clean_makeup_gain"] = makeup_report
    if makeup_report.get("applied"):
        report["actions"].append("pre_clean_makeup_gain_compensation")
    cleaned_before = _merge_fast_qc_into_analysis(before, cleaned_fast, label="post_residue_pre_clean")
    cleaned_before["original_input_analysis_before_pre_clean"] = copy.deepcopy(before)
    cleaned_before["residue_pre_clean_stage"] = report
    # Preserve the original Centrifuge evidence; it is the source detector that caused cleanup.
    if isinstance(before.get("frequency_band_centrifuge"), dict):
        cleaned_before["frequency_band_centrifuge"] = before.get("frequency_band_centrifuge")

    report["cleaned_metrics"] = {
        "integrated_lufs": cleaned_before.get("integrated_lufs"),
        "true_peak": cleaned_before.get("approx_true_peak_dbfs"),
        "crest": cleaned_before.get("crest_factor_db"),
        "correlation": cleaned_before.get("correlation"),
        "side_low": ((cleaned_before.get("ms") or {}) if isinstance(cleaned_before.get("ms"), dict) else {}).get("side_ratio_lowband"),
        "side_high": ((cleaned_before.get("ms") or {}) if isinstance(cleaned_before.get("ms"), dict) else {}).get("side_ratio_8k_14k"),
    }
    report["bypassed"] = False
    report["reason"] = "conditional_pre_clean_applied"
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report, cleaned_before

def _effective_oversample_factor(before: dict[str, Any] | None = None, decision: dict[str, Any] | None = None) -> tuple[int, dict[str, Any]]:
    """Return final limiter oversampling with adaptive crash guards.

    32x is allowed, but long Streamlit Cloud jobs are automatically capped unless
    BUSY_FORCE_OVERSAMPLE=1. This keeps the app usable while still allowing 16x/32x
    for shorter files or stronger servers.
    """
    raw = str(os.environ.get("BUSY_OVERSAMPLE", "8")).strip().lower()
    auto = raw in {"auto", "adaptive"}
    if auto:
        requested = 16
    else:
        try:
            requested = int(raw)
        except Exception:
            requested = 8
    requested = int(np.clip(requested, 2, 32))
    eff = requested
    before = before or {}
    duration = float(before.get("duration_sec", 0.0) or 0.0)
    artifact = before.get("suno_artifact_analysis", {}) if isinstance(before.get("suno_artifact_analysis", {}), dict) else {}
    artifact_risk = float(artifact.get("overall_risk", 0.0) or 0.0)
    force = _env_bool("BUSY_FORCE_OVERSAMPLE", False)
    adaptive = _env_bool("BUSY_ADAPTIVE_QUALITY", True)
    reason = "requested"
    two_stage = _env_bool("BUSY_TWO_STAGE_LIMITING", False)
    if auto:
        if two_stage and duration <= 300:
            eff, reason = 32, "two_stage_auto_short_or_medium_file"
        elif duration <= 180:
            eff, reason = 32, "auto_short_file"
        elif duration <= 330:
            eff, reason = 16, "auto_medium_file"
        else:
            eff, reason = 8, "auto_long_file"
        if artifact_risk >= 0.65 and duration <= 300:
            eff = max(eff, 16)
            reason += "+artifact_high"
    elif adaptive and not force:
        if requested >= 32 and two_stage and duration <= 420:
            eff, reason = 32, "32x_allowed_two_stage"
        elif requested >= 32 and duration > 300:
            eff, reason = 16, "32x_capped_long_file"
        elif requested >= 32 and duration > 210:
            eff, reason = 16, "32x_capped_streamlit_safe"
        elif requested >= 16 and duration > 420:
            eff, reason = 8, "16x_capped_very_long_file"
    return int(np.clip(eff, 2, 32)), {
        "requested_oversample": requested,
        "effective_oversample": int(np.clip(eff, 2, 32)),
        "adaptive_quality": adaptive,
        "force_oversample": force,
        "duration_sec": round(duration, 3),
        "artifact_risk": round(artifact_risk, 3),
        "reason": reason,
    }

def _rollback_max_attempts() -> int:
    try:
        val = int(os.environ.get("BUSY_ROLLBACK_MAX_ATTEMPTS", "2"))
    except Exception:
        val = 2
    return int(np.clip(val, 0, 3))


def _decision_is_role_first(decision: dict[str, Any] | None) -> bool:
    if not isinstance(decision, dict):
        return False
    if str(decision.get("actual_dsp_basis", "")).lower().startswith("role_first"):
        return True
    gmp = decision.get("genre_mix_policy") if isinstance(decision.get("genre_mix_policy"), dict) else {}
    if str(gmp.get("actual_dsp_basis", "")).lower().startswith("role_first"):
        return True
    flags = decision.get("processing_flags") or []
    return isinstance(flags, list) and any(str(x).startswith("role_first") for x in flags)


def _effective_rollback_max_attempts(y: np.ndarray, sr: int, decision: dict[str, Any] | None) -> int:
    max_attempts = _rollback_max_attempts()
    try:
        duration = float(getattr(y, "shape", [0])[0]) / max(1, int(sr))
    except Exception:
        duration = 0.0
    if str(os.environ.get("BUSY_LONG_TRACK_ROLLBACK_CAP", "1")).lower() in {"1", "true", "yes", "on"}:
        if duration >= float(os.environ.get("BUSY_LONG_TRACK_ROLLBACK_SEC", "240")):
            max_attempts = min(max_attempts, int(os.environ.get("BUSY_LONG_TRACK_ROLLBACK_MAX_ATTEMPTS", "1")))
        if duration >= float(os.environ.get("BUSY_VERY_LONG_TRACK_ROLLBACK_SEC", "420")):
            max_attempts = min(max_attempts, int(os.environ.get("BUSY_VERY_LONG_TRACK_ROLLBACK_MAX_ATTEMPTS", "0")))
    if _decision_is_role_first(decision):
        max_attempts = min(max_attempts, int(os.environ.get("BUSY_ROLE_FIRST_ROLLBACK_MAX_ATTEMPTS", "1")))
    return int(np.clip(max_attempts, 0, 3))


def _qc_oversample_factor() -> int:
    # Candidate QC does not need final 16x/32x precision. Final normalize still uses full BUSY_OVERSAMPLE.
    try:
        val = int(os.environ.get("BUSY_QC_OVERSAMPLE", "2"))
    except Exception:
        val = 2
    return int(np.clip(val, 1, 4))

def _approx_true_peak_db(y: np.ndarray, oversample: int | None = None) -> float:
    if y.size == 0:
        return -120.0
    os_factor = oversample or _oversample_factor()
    return true_peak_db_oversampled_chunked(y, oversample=os_factor, sr=48_000)


def _normalize_to_true_peak(y: np.ndarray, target_db: float = FINAL_TRUE_PEAK_CEILING_DB, sr: int = 48_000, oversample: int | None = None) -> tuple[np.ndarray, float]:
    os_factor = int(oversample or _oversample_factor())
    out, gain_db, _tp = normalize_to_true_peak_chunked(y, target_db=target_db, oversample=os_factor, sr=sr)
    return out, gain_db


def _attenuate_to_true_peak_ceiling(
    y: np.ndarray,
    target_db: float = FINAL_TRUE_PEAK_CEILING_DB,
    sr: int = 48_000,
    oversample: int | None = None,
    tolerance_db: float = 0.02,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """True-peak safety ceiling that never boosts loudness upward.

    Older engine paths used normalize_to_true_peak after the loudness iteration,
    which can add several dB when the signal's peak is below ceiling.  That turns
    a target/floor check into a hidden loudness push.  v62.9.3 keeps the ceiling
    stage attenuation-only: it reduces hot peaks, but does not raise safe audio to
    the ceiling.
    """
    os_factor = int(oversample or _oversample_factor())
    x = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    report: dict[str, Any] = {
        "schema_version": "busy_true_peak_ceiling_attenuation_only_v8_5_3_62_9_3",
        "target_db": round(float(target_db), 3),
        "oversample": int(os_factor),
        "policy": "true-peak ceiling is attenuation-only after loudness targeting; it must not boost a below-ceiling master",
    }
    try:
        tp = float(true_peak_db_oversampled_chunked(x, oversample=max(1, int(os_factor)), sr=sr, chunk_sec=12.0, overlap_sec=0.25))
        report["pre_true_peak_dbtp"] = round(tp, 3)
        if tp > float(target_db) + float(tolerance_db):
            gain_db = float(target_db) - tp
            out = apply_gain_db(x, gain_db)
            report.update({"active": True, "gain_db": round(float(gain_db), 3), "reason": "attenuated_to_true_peak_ceiling"})
            return out, float(gain_db), report
        report.update({"active": False, "gain_db": 0.0, "reason": "already_below_true_peak_ceiling_no_boost"})
        return x, 0.0, report
    except Exception as exc:
        report.update({"active": False, "gain_db": 0.0, "reason": "true_peak_measurement_exception_no_boost", "error": str(exc)[:240]})
        return x, 0.0, report




def _v48_2_2_export_lock_delta(expected: dict[str, Any] | None, observed: dict[str, Any] | None) -> dict[str, Any]:
    """Compare the selected candidate analysis with the actual returned buffer.

    This is a buffer-level export lock foundation.  The worker/output layer can
    add the disk SHA256 identity check, but the mastering engine must already
    verify that the exact audio it is about to return has not drifted away from
    the selected candidate's final-grade analysis.
    """
    expected = expected if isinstance(expected, dict) else {}
    observed = observed if isinstance(observed, dict) else {}
    def val(d: dict[str, Any], *keys: str, default: float = -999.0) -> float:
        for k in keys:
            if k in d:
                try:
                    out = float(d.get(k))
                    if np.isfinite(out):
                        return float(out)
                except Exception:
                    pass
        return float(default)
    exp_lufs = val(expected, "integrated_lufs", "lufs")
    obs_lufs = val(observed, "integrated_lufs", "lufs")
    exp_tp = val(expected, "approx_true_peak_dbfs", "true_peak_dbtp", "true_peak")
    obs_tp = val(observed, "approx_true_peak_dbfs", "true_peak_dbtp", "true_peak")
    exp_crest = val(expected, "crest_factor_db", "crest_db", default=999.0)
    obs_crest = val(observed, "crest_factor_db", "crest_db", default=999.0)
    exp_lra = val(expected, "lra_lu", "loudness_range_lu", default=0.0)
    obs_lra = val(observed, "lra_lu", "loudness_range_lu", default=0.0)
    return {
        "lufs_delta": round(float(obs_lufs - exp_lufs), 3) if exp_lufs > -998 and obs_lufs > -998 else None,
        "true_peak_delta_db": round(float(obs_tp - exp_tp), 3) if exp_tp > -998 and obs_tp > -998 else None,
        "crest_delta_db": round(float(obs_crest - exp_crest), 3) if exp_crest < 998 and obs_crest < 998 else None,
        "lra_delta_lu": round(float(obs_lra - exp_lra), 3) if exp_lra > -998 and obs_lra > -998 else None,
        "expected": {"integrated_lufs": exp_lufs, "true_peak": exp_tp, "crest": exp_crest, "lra": exp_lra},
        "observed": {"integrated_lufs": obs_lufs, "true_peak": obs_tp, "crest": obs_crest, "lra": obs_lra},
    }




def _v56_audio_buffer_identity(
    y: np.ndarray,
    sr: int,
    analysis: dict[str, Any] | None = None,
    *,
    stage: str = "",
    source: str = "",
    candidate_label: str | None = None,
) -> dict[str, Any]:
    """Deterministic lightweight identity for the in-memory audio buffer.

    The full WAV/file SHA can only be known by the export/storage layer.  This
    engine-level identity pins the buffer that was analysed and handed off so
    selected-candidate analysis, export-lock analysis, and the delivered buffer
    are no longer compared across mixed stages or mixed analyzers.
    """
    out: dict[str, Any] = {
        "schema_version": "busy_audio_buffer_identity_v8_5_3_56",
        "stage": str(stage or "")[:120],
        "source": str(source or "")[:160],
        "candidate_label": str(candidate_label or "")[:180] or None,
        "sample_rate": int(sr) if sr else None,
    }
    try:
        arr = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        arr_c = np.ascontiguousarray(arr.astype("<f4", copy=False))
        digest = hashlib.sha256(arr_c.tobytes()).hexdigest()
        out.update({
            "sha256": digest,
            "sha256_16": digest[:16],
            "shape": list(arr_c.shape),
            "dtype": "float32_le",
            "duration_sec": round(float(arr_c.shape[0]) / max(1, int(sr)), 3) if arr_c.ndim >= 1 else None,
            "peak_abs": round(float(np.max(np.abs(arr_c))) if arr_c.size else 0.0, 8),
            "rms": round(float(np.sqrt(np.mean(np.square(arr_c)))) if arr_c.size else 0.0, 8),
        })
        if isinstance(analysis, dict) and analysis:
            out["analysis_summary"] = _analysis_summary_for_loudness(analysis)
    except Exception as exc:
        out.update({"error": str(exc)[:240]})
    return out


def _v56_selected_export_expected_analysis(selected_render_report: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """Return selected-candidate expected analysis using the same analyzer family as final export analysis.

    Earlier patches stored selected candidate post-analysis from fast QC, then
    compared it to final full `analyze_audio()` output.  On the user's v55.3.2
    test this caused false LUFS/TP/LRA mismatches even though the selected
    candidate lineage was still valid.  v56 prefers a selected-candidate full
    export analysis when available and falls back to fast QC only for older rows.
    """
    rep = selected_render_report if isinstance(selected_render_report, dict) else {}
    use_full = _env_bool("BUSY_V56_EXPORT_LOCK_USE_FULL_SELECTED_ANALYSIS", True)
    full = rep.get("post_analysis_full") if isinstance(rep.get("post_analysis_full"), dict) else {}
    if use_full and full:
        return full, "full_analyze_audio"
    fast = rep.get("post_analysis") if isinstance(rep.get("post_analysis"), dict) else {}
    return fast if isinstance(fast, dict) else {}, "fast_qc_legacy_fallback"

def _v48_2_2_export_lock_mismatch(delta: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    lufs_tol = _env_float("BUSY_V48_2_2_EXPORT_LOCK_LUFS_TOL", 0.30, 0.05, 2.0)
    tp_tol = _env_float("BUSY_V48_2_2_EXPORT_LOCK_TP_TOL", 0.30, 0.05, 2.0)
    crest_tol = _env_float("BUSY_V48_2_2_EXPORT_LOCK_CREST_TOL", 0.50, 0.05, 3.0)
    lra_tol = _env_float("BUSY_V48_2_2_EXPORT_LOCK_LRA_TOL", 0.60, 0.05, 4.0)
    try:
        if delta.get("lufs_delta") is not None and abs(float(delta.get("lufs_delta"))) > lufs_tol:
            reasons.append("lufs_mismatch")
        if delta.get("true_peak_delta_db") is not None and abs(float(delta.get("true_peak_delta_db"))) > tp_tol:
            reasons.append("true_peak_mismatch")
        if delta.get("crest_delta_db") is not None and abs(float(delta.get("crest_delta_db"))) > crest_tol:
            reasons.append("crest_mismatch")
        if delta.get("lra_delta_lu") is not None and abs(float(delta.get("lra_delta_lu"))) > lra_tol:
            reasons.append("lra_mismatch")
    except Exception:
        reasons.append("export_lock_delta_parse_error")
    return reasons



# ---------------------------------------------------------------------------
# v8.5.3.60: Deterministic Multi-Stage Limiter Foundation.
# ---------------------------------------------------------------------------

def _v60_float_metric(analysis: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    if not isinstance(analysis, dict):
        return default
    for key in keys:
        try:
            value = analysis.get(key)
            if value is not None:
                out = float(value)
                if np.isfinite(out):
                    return float(out)
        except Exception:
            pass
    if any("lra" in str(k).lower() for k in keys):
        try:
            val = _analysis_lra_lu(analysis)
            if val is not None:
                return float(val)
        except Exception:
            pass
    return default




V6362_DML_SCHEMA_VERSION = "busy_v6362_dml_ownership_contract_stage1"
V6362_DML_ENGINE_VERSION = "busy_auto_mastering_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
V6362_MASTERING_ENGINE_VERSION = "busy_auto_mastering_v8_5_3_64_0_4_commercial_maximum_density_push_foundation"


def _v6362_ensure_lra_for_dml(
    analysis: dict[str, Any] | None,
    audio: np.ndarray | None,
    sr: int,
    *,
    source: str,
) -> dict[str, Any]:
    """Attach same-method short-term LRA proxy when fast-QC omits LRA.

    This is deliberately limited to DML candidate/contract telemetry.  It does
    not replace final-report official LRA authority and it does not change QC
    warning thresholds.  The only goal is to keep standard-vs-DML and
    input-relative DML damage guards from comparing ``None`` values.
    """
    out: dict[str, Any] = copy.deepcopy(analysis) if isinstance(analysis, dict) else {}
    try:
        existing = _analysis_lra_lu(out)
        if existing is not None and np.isfinite(float(existing)):
            out.setdefault("lra_lu", round(float(existing), 3))
            out.setdefault("lra_lu_authority", out.get("lra_lu_authority") or "analysis_report_or_analyzer")
            return out
    except Exception:
        pass
    if audio is None:
        return out
    try:
        proxy = _runtime_safe_lra_summary(audio, sr)
        val = proxy.get("lra_lu") if isinstance(proxy, dict) else None
        if val is None or not np.isfinite(float(val)):
            return out
        loud = dict(out.get("loudness") or {}) if isinstance(out.get("loudness"), dict) else {}
        st = dict(loud.get("short_term_lufs") or {}) if isinstance(loud.get("short_term_lufs"), dict) else {}
        st.update(proxy)
        st["lra_lu"] = round(float(val), 3)
        st["lra_authority"] = "runtime_safe_short_term_lra_proxy_v6362_dml"
        st["proxy_source"] = source
        st["policy"] = "Same-method DML-only LRA proxy for candidate guard/telemetry; final report LRA authority remains unchanged."
        loud["short_term_lufs"] = st
        out["loudness"] = loud
        out["lra_lu"] = round(float(val), 3)
        out["lra_lu_authority"] = "runtime_safe_short_term_lra_proxy_v6362_dml"
        out["lra_proxy_source"] = source
    except Exception as exc:
        out.setdefault("v6362_lra_proxy_error", str(exc)[:180])
    return out


def _v6362_num(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return default


def _v6362_metric(analysis: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    analysis = analysis if isinstance(analysis, dict) else {}
    for key in keys:
        val = _v6362_num(analysis.get(key), None)
        if val is not None:
            return val
    loud = analysis.get("loudness") if isinstance(analysis.get("loudness"), dict) else {}
    for key in keys:
        val = _v6362_num(loud.get(key), None)
        if val is not None:
            return val
    if any("lra" in str(k).lower() for k in keys):
        val = _analysis_lra_lu(analysis)
        if val is not None:
            return _v6362_num(val, default)
    return default


def _v6362_lra_authority(analysis: dict[str, Any] | None) -> str | None:
    a = analysis if isinstance(analysis, dict) else {}
    loud = a.get("loudness") if isinstance(a.get("loudness"), dict) else {}
    st = loud.get("short_term_lufs") if isinstance(loud.get("short_term_lufs"), dict) else {}
    return a.get("lra_lu_authority") or st.get("lra_authority")


def _v6362_damage_class(metric: str, input_value: float | None, delta: float | None) -> str:
    """Machine-readable input-relative damage class for DML contract telemetry."""
    if input_value is None or delta is None:
        return "unknown"
    loss = max(0.0, -float(delta))
    if metric == "lra":
        iv = float(input_value)
        if iv >= 8.0:
            bands = (1.0, 2.0, 3.0, 4.0, 4.5)
        elif iv >= 4.0:
            bands = (0.8, 1.5, 2.5, 3.5, 4.0)
        elif iv >= 2.0:
            bands = (0.4, 0.8, 1.2, 1.8, 2.5)
        elif iv >= 1.5:
            bands = (0.3, 0.5, 0.8, 1.2, 1.5)
        else:
            bands = (0.0, 0.2, 0.4, 0.6, 0.8)
    else:
        iv = float(input_value)
        if iv >= 12.0:
            bands = (0.7, 1.2, 1.8, 2.5, 3.5)
        elif iv >= 9.0:
            bands = (0.5, 1.0, 1.5, 2.0, 2.5)
        elif iv >= 7.0:
            bands = (0.4, 0.7, 1.2, 1.6, 2.0)
        else:
            bands = (0.3, 0.5, 0.8, 1.2, 1.5)
    labels = ("safe", "acceptable", "caution", "warning", "high_risk", "reject")
    for i, limit in enumerate(bands):
        if loss <= float(limit) + 1e-9:
            return labels[i]
    return labels[-1]


def _v6362_dml_ownership_contract(
    before: dict[str, Any] | None,
    standard_analysis: dict[str, Any] | None,
    candidate_analysis: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    finish_report: dict[str, Any] | None,
    limiter_budget_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    """v63.6.2 stage-1 contract tying ownership, DML and target-miss policy.

    This is a guard/telemetry contract, not a louder limiter.  It makes the
    finalizer explicitly inherit v63.5.1 ownership authority, caps any DML LUFS
    change inside the allowed final-push budget, and records when target/floor
    misses are accepted because TP, artifact or input-relative damage limits own
    the decision.
    """
    dec = decision if isinstance(decision, dict) else {}
    fin = finish_report if isinstance(finish_report, dict) else {}
    before = before if isinstance(before, dict) else {}
    std = standard_analysis if isinstance(standard_analysis, dict) else {}
    cand = candidate_analysis if isinstance(candidate_analysis, dict) else {}
    plan = limiter_budget_plan if isinstance(limiter_budget_plan, dict) else {}
    ownership = plan.get("mastering_ownership") if isinstance(plan.get("mastering_ownership"), dict) else _v6351_find_existing_ownership(dec, fin)
    ownership = ownership if isinstance(ownership, dict) else {}
    hp = _bamix_handoff_policy_from_decision(dec)
    hp = hp if isinstance(hp, dict) else {}
    targets = dec.get("targets") if isinstance(dec.get("targets"), dict) else {}
    target = _v6351_first_num(hp.get("effective_final_target_lufs"), hp.get("preferred_final_target_lufs"), fin.get("target_lufs"), targets.get("bamix_mastering_target_lufs"), targets.get("integrated_lufs_target"), default=None)
    floor = _v6351_first_num(hp.get("effective_floor_lufs"), fin.get("commercial_floor_lufs"), default=(float(target) - 0.75 if target is not None else None))
    std_lufs = _v6362_metric(std, "integrated_lufs", "stereo_integrated_lufs", default=None)
    cand_lufs = _v6362_metric(cand, "integrated_lufs", "stereo_integrated_lufs", default=None)
    std_tp = _v6362_metric(std, "approx_true_peak_dbfs", "true_peak_dbtp", "true_peak_dbfs", default=None)
    cand_tp = _v6362_metric(cand, "approx_true_peak_dbfs", "true_peak_dbtp", "true_peak_dbfs", default=None)
    std_lra = _v6362_metric(std, "lra_lu", "loudness_range_lu", default=None)
    cand_lra = _v6362_metric(cand, "lra_lu", "loudness_range_lu", default=None)
    std_crest = _v6362_metric(std, "crest_factor_db", "crest_db", default=None)
    cand_crest = _v6362_metric(cand, "crest_factor_db", "crest_db", default=None)
    input_lra = _analysis_lra_lu(before) if before else None
    if input_lra is None:
        input_lra = _v6351_first_num(hp.get("premaster_lra_lu"), default=None)
    input_crest = _v6351_first_num(before.get("crest_factor_db") if before else None, before.get("crest_db") if before else None, hp.get("premaster_crest_factor_db"), default=None)
    lufs_delta = (float(cand_lufs) - float(std_lufs)) if cand_lufs is not None and std_lufs is not None else None
    dml_lra_delta = (float(cand_lra) - float(std_lra)) if cand_lra is not None and std_lra is not None else None
    dml_crest_delta = (float(cand_crest) - float(std_crest)) if cand_crest is not None and std_crest is not None else None
    input_to_candidate_lra_delta = (float(cand_lra) - float(input_lra)) if cand_lra is not None and input_lra is not None else None
    input_to_candidate_crest_delta = (float(cand_crest) - float(input_crest)) if cand_crest is not None and input_crest is not None else None
    input_to_standard_lra_delta = (float(std_lra) - float(input_lra)) if std_lra is not None and input_lra is not None else None
    input_to_standard_crest_delta = (float(std_crest) - float(input_crest)) if std_crest is not None and input_crest is not None else None
    allowed_push = _v6351_first_num(ownership.get("allowed_final_push_db"), ownership.get("final_push_budget_db"), default=0.0)
    authority = str(ownership.get("push_authority") or "unknown")
    selected_class = str(ownership.get("selected_loudness_class") or "unknown")
    workload_cap = _v6351_first_num(ownership.get("limiter_workload_budget_lu"), ownership.get("max_limiter_workload_lu"), default=None)
    estimated_workload = _v6351_first_num((ownership.get("ownership_inputs") or {}).get("estimated_final_limiter_workload_lu") if isinstance(ownership.get("ownership_inputs"), dict) else None, plan.get("estimated_premaster_to_final_workload_lu"), default=None)
    tp_ceiling = _v6351_first_num((ownership.get("tp_ceiling_policy") or {}).get("ceiling_dbtp") if isinstance(ownership.get("tp_ceiling_policy"), dict) else None, hp.get("true_peak_ceiling_dbtp"), default=float(FINAL_TRUE_PEAK_CEILING_DB))
    tp_room_after = (float(tp_ceiling) - float(cand_tp)) if tp_ceiling is not None and cand_tp is not None else None
    target_gap_after = max(0.0, float(target) - float(cand_lufs)) if target is not None and cand_lufs is not None else None
    floor_gap_after = max(0.0, float(floor) - float(cand_lufs)) if floor is not None and cand_lufs is not None else None
    workload_excess = (float(estimated_workload) - float(workload_cap)) if estimated_workload is not None and workload_cap is not None else None
    reject_reasons: list[str] = []
    notes: list[str] = []
    tol = _env_float("BUSY_V6362_DML_OWNERSHIP_PUSH_TOL_DB", 0.05, 0.0, 0.4)
    if lufs_delta is not None and allowed_push is not None and float(lufs_delta) > float(allowed_push) + float(tol):
        reject_reasons.append("v6362_dml_candidate_exceeds_final_push_budget")
    if authority == "artifact_protected_fallback" and lufs_delta is not None and float(lufs_delta) > _env_float("BUSY_V6362_ARTIFACT_FALLBACK_MAX_DML_DELTA_DB", 0.03, -0.2, 0.4):
        reject_reasons.append("v6362_artifact_protected_fallback_blocks_dml_loudness_gain")
    if dml_lra_delta is not None and float(dml_lra_delta) < -_env_float("BUSY_V6362_DML_ADDED_LRA_LOSS_MAX_LU", 0.20, 0.0, 1.0):
        reject_reasons.append("v6362_dml_added_lra_loss_exceeds_contract")
    if dml_crest_delta is not None and float(dml_crest_delta) < -_env_float("BUSY_V6362_DML_ADDED_CREST_LOSS_MAX_DB", 0.35, 0.0, 2.0):
        reject_reasons.append("v6362_dml_added_crest_loss_exceeds_contract")
    if cand_tp is not None and tp_ceiling is not None and float(cand_tp) > float(tp_ceiling) + _env_float("BUSY_V6362_DML_TP_TOL_DB", 0.03, 0.0, 0.2):
        reject_reasons.append("v6362_dml_candidate_exceeds_locked_tp_ceiling")
    target_miss_allowed = False
    if target_gap_after is not None and target_gap_after > _env_float("BUSY_V6362_TARGET_MISS_MIN_GAP_LU", 0.05, 0.0, 1.0):
        if authority in {"push_limited", "artifact_protected_fallback"} or (workload_excess is not None and workload_excess > 0.0) or (tp_room_after is not None and tp_room_after < 0.12):
            target_miss_allowed = True
            notes.append("target_miss_allowed_by_ownership_or_safety_budget")
        else:
            notes.append("target_miss_present_but_not_forced_by_dml")
    if workload_excess is not None and workload_excess > 0.0:
        notes.append("final_limiter_workload_cap_exceeded_upstream_density_required")
    return {
        "schema_version": V6362_DML_SCHEMA_VERSION,
        "engine_version": V6362_DML_ENGINE_VERSION,
        "active": True,
        "route": "mastering_ownership_dml_contract",
        "ownership_source": "v6350_limiter_workload_budget_plan.mastering_ownership" if isinstance(plan.get("mastering_ownership"), dict) else "v6351_existing_ownership",
        "selected_loudness_class": selected_class,
        "push_authority": authority,
        "allowed_final_push_db": round(float(allowed_push), 3) if allowed_push is not None else None,
        "limiter_workload_budget_lu": round(float(workload_cap), 3) if workload_cap is not None else None,
        "estimated_final_limiter_workload_lu": round(float(estimated_workload), 3) if estimated_workload is not None else None,
        "workload_excess_over_cap_lu": round(float(workload_excess), 3) if workload_excess is not None else None,
        "target_lufs": round(float(target), 3) if target is not None else None,
        "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
        "target_gap_after_dml_lu": round(float(target_gap_after), 3) if target_gap_after is not None else None,
        "floor_gap_after_dml_lu": round(float(floor_gap_after), 3) if floor_gap_after is not None else None,
        "target_miss_allowed": bool(target_miss_allowed),
        "target_miss_policy": "target_lufs_is_reference_not_command; miss is allowed when ownership, TP room, artifact fallback, or input-relative damage budget says stop",
        "metrics": {
            "standard_lufs": round(float(std_lufs), 3) if std_lufs is not None else None,
            "candidate_lufs": round(float(cand_lufs), 3) if cand_lufs is not None else None,
            "dml_lufs_delta_db": round(float(lufs_delta), 3) if lufs_delta is not None else None,
            "standard_lra_lu": round(float(std_lra), 3) if std_lra is not None else None,
            "candidate_lra_lu": round(float(cand_lra), 3) if cand_lra is not None else None,
            "dml_lra_delta_lu": round(float(dml_lra_delta), 3) if dml_lra_delta is not None else None,
            "standard_lra_authority": _v6362_lra_authority(std),
            "candidate_lra_authority": _v6362_lra_authority(cand),
            "standard_crest_db": round(float(std_crest), 3) if std_crest is not None else None,
            "candidate_crest_db": round(float(cand_crest), 3) if cand_crest is not None else None,
            "dml_crest_delta_db": round(float(dml_crest_delta), 3) if dml_crest_delta is not None else None,
            "candidate_true_peak_dbtp": round(float(cand_tp), 3) if cand_tp is not None else None,
            "tp_room_after_dml_db": round(float(tp_room_after), 3) if tp_room_after is not None else None,
        },
        "input_relative_damage_ownership": {
            "input_lra_lu": round(float(input_lra), 3) if input_lra is not None else None,
            "standard_lra_delta_lu": round(float(input_to_standard_lra_delta), 3) if input_to_standard_lra_delta is not None else None,
            "candidate_lra_delta_lu": round(float(input_to_candidate_lra_delta), 3) if input_to_candidate_lra_delta is not None else None,
            "candidate_lra_class": _v6362_damage_class("lra", input_lra, input_to_candidate_lra_delta),
            "input_crest_db": round(float(input_crest), 3) if input_crest is not None else None,
            "standard_crest_delta_db": round(float(input_to_standard_crest_delta), 3) if input_to_standard_crest_delta is not None else None,
            "candidate_crest_delta_db": round(float(input_to_candidate_crest_delta), 3) if input_to_candidate_crest_delta is not None else None,
            "candidate_crest_class": _v6362_damage_class("crest", input_crest, input_to_candidate_crest_delta),
            "dml_added_damage_only": True,
            "policy": "If damage is inherited from the standard final, DML records it and may allow target miss; only DML-added damage can reject the DML candidate here.",
        },
        "reject_reasons": reject_reasons,
        "notes": notes,
        "contract_passed": bool(not reject_reasons),
        "forbidden_actions": [
            "do_not_raise_true_peak_ceiling",
            "do_not_increase_limiter_push_to_hit_target",
            "do_not_hide_warning_labels",
            "do_not_solve_low_mid_bottleneck_inside_dml",
        ],
        "policy": "v63.6.2 stage-1: ownership, DML and final limiter workload are a single contract; DML may polish/catch peaks only inside ownership budget, and target miss is explicit when safety owns the stop condition.",
    }

def _v60_candidate_acceptance(
    before: dict[str, Any] | None,
    standard_analysis: dict[str, Any] | None,
    candidate_analysis: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    finish_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict, damage-aware gate for v60 DML controlled activation.

    Target LUFS is intentionally not a command here.  The DML candidate may
    replace the legacy final only if it is at least analysis-equivalent and does
    not lose LRA/crest or introduce new safety problems.  Otherwise it remains a
    shadow candidate for telemetry/learning.
    """
    std = standard_analysis if isinstance(standard_analysis, dict) else {}
    cand = candidate_analysis if isinstance(candidate_analysis, dict) else {}
    std_lufs = _v60_float_metric(std, "integrated_lufs", default=-999.0)
    cand_lufs = _v60_float_metric(cand, "integrated_lufs", default=-999.0)
    std_tp = _v60_float_metric(std, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
    cand_tp = _v60_float_metric(cand, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
    std_lra = _v60_float_metric(std, "lra_lu", "loudness_range_lu", default=None)
    cand_lra = _v60_float_metric(cand, "lra_lu", "loudness_range_lu", default=None)
    std_crest = _v60_float_metric(std, "crest_factor_db", default=None)
    cand_crest = _v60_float_metric(cand, "crest_factor_db", default=None)
    lra_loss = (float(cand_lra) - float(std_lra)) if cand_lra is not None and std_lra is not None else None
    crest_loss = (float(cand_crest) - float(std_crest)) if cand_crest is not None and std_crest is not None else None
    lufs_delta = float(cand_lufs) - float(std_lufs) if cand_lufs is not None and std_lufs is not None else None
    tp_delta = float(cand_tp) - float(std_tp) if cand_tp is not None and std_tp is not None else None
    # v63.1.6.2: deep-research alignment.  v60 is a final-stage architecture,
    # not merely a louder-than-standard candidate.  A safe, analysis-equivalent
    # multistage finalizer may replace the standard final even when LUFS delta is
    # near zero; loudness push budget must not block tonal/density/finalizer QC.
    deep_align = _env_bool("BUSY_V63162_DEEP_RESEARCH_ALIGNMENT", True)
    max_lra_loss = _env_float("BUSY_V60_MAX_LRA_LOSS_LU", 0.20 if deep_align else 0.12, 0.0, 1.0)
    max_crest_loss = _env_float("BUSY_V60_MAX_CREST_LOSS_DB", 0.35 if deep_align else 0.20, 0.0, 2.0)
    min_lufs_delta = _env_float("BUSY_V60_MIN_LUFS_DELTA_TO_APPLY", -0.18 if deep_align else 0.03, -1.0, 1.0)
    reasons: list[str] = []
    governance_notes: list[str] = []
    try:
        if cand_tp is not None and float(cand_tp) > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.03:
            reasons.append("true_peak_over_ceiling")
        if lra_loss is not None and float(lra_loss) < -max_lra_loss:
            reasons.append("lra_loss_exceeds_v60_gate")
        if crest_loss is not None and float(crest_loss) < -max_crest_loss:
            reasons.append("crest_loss_exceeds_v60_gate")
        if lufs_delta is not None and float(lufs_delta) < min_lufs_delta:
            reasons.append("dml_candidate_too_quiet_vs_standard_final")
        elif deep_align and lufs_delta is not None and float(lufs_delta) < 0.03:
            governance_notes.append("analysis_equivalent_loudness_allowed_for_multistage_finalizer")
        max_lufs_delta = _env_float("BUSY_V60_DML_MAX_LUFS_DELTA_DB", 0.75, 0.05, 3.0)
        if lufs_delta is not None and float(lufs_delta) > max_lufs_delta and not _env_bool("BUSY_V60_DML_ALLOW_LARGE_LUFS_DELTA", False):
            reasons.append("dml_loudness_delta_exceeds_runtime_gate")
        finish_report = finish_report if isinstance(finish_report, dict) else {}
        ownership = _v6351_find_existing_ownership(decision if isinstance(decision, dict) else {}, finish_report)
        if isinstance(ownership, dict) and ownership.get("active"):
            try:
                allowed_delta = float(ownership.get("allowed_final_push_db", ownership.get("final_push_budget_db", 0.0)) or 0.0)
            except Exception:
                allowed_delta = 0.0
            authority = str(ownership.get("push_authority") or "")
            if lufs_delta is not None and float(lufs_delta) > allowed_delta + _env_float("BUSY_V6351_DML_OWNERSHIP_DELTA_TOL_DB", 0.08, 0.0, 0.5):
                reasons.append("v6351_candidate_exceeds_ownership_push_budget")
            if authority == "artifact_protected_fallback" and lufs_delta is not None and float(lufs_delta) > _env_float("BUSY_V6351_ARTIFACT_FALLBACK_MAX_LUFS_DELTA_DB", 0.03, -0.2, 0.4):
                reasons.append("v6351_artifact_protected_blocks_loudness_gain")
            if authority == "push_limited":
                governance_notes.append("v6351_push_limited_budget_respected_by_dml_acceptance")
            elif authority == "artifact_protected_fallback":
                governance_notes.append("v6351_artifact_protected_target_miss_allowed")
        target_lufs = _v60_float_metric(finish_report, "target_lufs", default=None)
        floor_lufs = _v60_float_metric(finish_report, "commercial_floor_lufs", default=None)
        if target_lufs is None and isinstance(decision, dict):
            targets = decision.get("targets") if isinstance(decision.get("targets"), dict) else {}
            target_lufs = _v60_float_metric(targets, "target_lufs", "integrated_lufs_target", default=None)
        overshoot_tol = _env_float("BUSY_V60_DML_TARGET_OVERSHOOT_TOLERANCE_DB", 0.45, 0.0, 2.0)
        if cand_lufs is not None and target_lufs is not None and float(cand_lufs) > float(target_lufs) + overshoot_tol:
            reasons.append("dml_candidate_overshoots_target_lufs")
        elif cand_lufs is not None and floor_lufs is not None and float(cand_lufs) > float(floor_lufs) + _env_float("BUSY_V60_DML_FLOOR_OVERSHOOT_TOLERANCE_DB", 1.25, 0.2, 3.0):
            reasons.append("dml_candidate_overshoots_commercial_floor_margin")
        feasibility_sources: list[tuple[str, dict[str, Any]]] = []
        v57 = finish_report.get("v57_commercial_loudness_feasibility") if isinstance(finish_report.get("v57_commercial_loudness_feasibility"), dict) else {}
        if v57:
            feasibility_sources.append(("v57", v57))
        v628 = {}
        try:
            prep = (decision or {}).get("v62_8_mastering_blocker_preparation") if isinstance(decision, dict) else {}
            if isinstance(prep, dict) and isinstance(prep.get("commercial_loudness_feasibility"), dict):
                v628 = prep.get("commercial_loudness_feasibility") or {}
        except Exception:
            v628 = {}
        if v628:
            feasibility_sources.append(("v62_8", v628))
        for label, feasibility in feasibility_sources:
            safe_budget = _v60_float_metric(feasibility, "safe_push_budget_db", default=None)
            feas_reasons = [str(x) for x in (feasibility.get("reasons") or []) if x]
            feas_decision = str(feasibility.get("decision") or "")
            if safe_budget is not None and float(safe_budget) <= 0.001:
                risky = any(("lra" in r or "crest" in r or "transient" in r or "true_peak" in r) for r in feas_reasons) or "stop_or_prepare" in feas_decision or "do_not_push" in feas_decision
                if risky:
                    if deep_align:
                        governance_notes.append(f"{label}_safe_push_budget_zero_blocks_loudness_push_only")
                    else:
                        reasons.append(f"{label}_safe_push_budget_zero_respected")
        std_risk = _safety_risk(before or {}, std, decision or {}) if std else []
        cand_risk = _safety_risk(before or {}, cand, decision or {}) if cand else []
        new_risk = [r for r in cand_risk if r not in set(std_risk)]
        hard_new = [r for r in new_risk if r in {"true_peak_too_hot", "phase_collapse", "low_end_collapse", "vocal_distortion", "lra_too_flat"}]
        if hard_new:
            reasons.append("new_hard_safety_risk")
        try:
            _v6362_plan = {}
            if isinstance(finish_report, dict) and isinstance(finish_report.get("v6350_limiter_workload_budget_plan"), dict):
                _v6362_plan = finish_report.get("v6350_limiter_workload_budget_plan") or {}
            _v6362_contract = _v6362_dml_ownership_contract(before, std, cand, decision, finish_report, _v6362_plan)
            for _r in (_v6362_contract.get("reject_reasons") or []):
                if _r not in reasons:
                    reasons.append(str(_r))
        except Exception as _v6362_exc:
            _v6362_contract = {"schema_version": V6362_DML_SCHEMA_VERSION, "engine_version": V6362_DML_ENGINE_VERSION, "active": False, "reason": "exception", "error": str(_v6362_exc)[:220]}
            reasons.append("v6362_contract_exception")
    except Exception as exc:
        reasons.append("v60_acceptance_exception:" + str(exc)[:80])
        _v6362_contract = {"schema_version": V6362_DML_SCHEMA_VERSION, "engine_version": V6362_DML_ENGINE_VERSION, "active": False, "reason": "outer_exception"}
    return {
        "schema_version": "busy_v6362_dml_acceptance_gate_ownership_contract",
        "engine_version": V6362_DML_ENGINE_VERSION,
        "accepted": bool(not reasons),
        "reasons": reasons,
        "metrics": {
            "standard_lufs": round(float(std_lufs), 3) if std_lufs is not None and std_lufs > -998 else None,
            "candidate_lufs": round(float(cand_lufs), 3) if cand_lufs is not None and cand_lufs > -998 else None,
            "lufs_delta_db": round(float(lufs_delta), 3) if lufs_delta is not None else None,
            "standard_lra_lu": round(float(std_lra), 3) if std_lra is not None else None,
            "candidate_lra_lu": round(float(cand_lra), 3) if cand_lra is not None else None,
            "lra_delta_lu": round(float(lra_loss), 3) if lra_loss is not None else None,
            "standard_crest_db": round(float(std_crest), 3) if std_crest is not None else None,
            "candidate_crest_db": round(float(cand_crest), 3) if cand_crest is not None else None,
            "crest_delta_db": round(float(crest_loss), 3) if crest_loss is not None else None,
            "standard_true_peak_dbtp": round(float(std_tp), 3) if std_tp is not None else None,
            "candidate_true_peak_dbtp": round(float(cand_tp), 3) if cand_tp is not None else None,
            "true_peak_delta_db": round(float(tp_delta), 3) if tp_delta is not None else None,
        },
        "v6362_dml_ownership_contract": _v6362_contract if '_v6362_contract' in locals() and isinstance(_v6362_contract, dict) else {"schema_version": V6362_DML_SCHEMA_VERSION, "active": False, "reason": "not_built"},
        "deep_research_alignment": {
            "schema_version": "busy_v6362_v60_deep_research_alignment",
            "active": bool(deep_align),
            "governance_notes": governance_notes,
            "v6351_mastering_ownership": ownership if 'ownership' in locals() and isinstance(ownership, dict) else {},
            "v6362_contract_active": bool(_v6362_contract.get("active")) if '_v6362_contract' in locals() and isinstance(_v6362_contract, dict) else False,
            "policy": "Safe-push budget controls loudness push only; finalizer/polish candidates are judged after safety limiting by post-QC, not rejected merely for analysis-equivalent LUFS. v63.6.2 additionally connects target-miss, ownership and input-relative DML damage into one contract.",
        },
        "policy": "DML replaces the existing final only when it is a bounded post-safety finalizer: no excessive LUFS jump, no target/floor overshoot, no DML-added LRA/crest/TP loss, and no new hard safety risks. v63.6.2 allows explicit target miss when ownership/artifact/TP/input-relative damage budgets say stop.",
    }



def _v6351_num(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return default


def _v6351_first_num(*values: Any, default: float | None = None) -> float | None:
    for value in values:
        out = _v6351_num(value, None)
        if out is not None:
            return out
    return default


def _v6351_list_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            if isinstance(v, (str, int, float, bool)):
                parts.append(f"{k}:{v}")
            elif isinstance(v, list):
                parts.extend(str(x) for x in v[:8] if isinstance(x, (str, int, float)))
        return " ".join(parts).lower()
    if isinstance(value, list):
        return " ".join(str(x) for x in value[:16]).lower()
    return str(value or "").lower()


def _v6351_find_existing_ownership(decision: dict[str, Any] | None = None, finish_report: dict[str, Any] | None = None, dml_report: dict[str, Any] | None = None) -> dict[str, Any]:
    for src in (finish_report, decision, dml_report):
        if not isinstance(src, dict):
            continue
        own = src.get("mastering_ownership")
        if isinstance(own, dict) and own.get("schema_version"):
            return own
        gate = src.get("commercial_push_safety_gate")
        if isinstance(gate, dict):
            own = gate.get("mastering_ownership")
            if isinstance(own, dict) and own.get("schema_version"):
                return own
        plan = src.get("v6350_limiter_workload_budget_plan")
        if isinstance(plan, dict):
            own = plan.get("mastering_ownership")
            if isinstance(own, dict) and own.get("schema_version"):
                return own
        v60 = src.get("v60_deterministic_multistage_limiter")
        if isinstance(v60, dict):
            plan = v60.get("v6350_limiter_workload_budget_plan")
            if isinstance(plan, dict):
                own = plan.get("mastering_ownership")
                if isinstance(own, dict) and own.get("schema_version"):
                    return own
    return {}


def _v6351_mastering_ownership_decision(
    decision: dict[str, Any] | None,
    current_analysis: dict[str, Any] | None = None,
    finish_report: dict[str, Any] | None = None,
    before_analysis: dict[str, Any] | None = None,
    *,
    stage_label: str = "finalizer",
) -> dict[str, Any]:
    """v63.5.1 Mastering Ownership / Commercial Push Safety Gate.

    This is a decision layer, not a new DSP stage.  It classifies who owns the
    last loudness push and returns bounded authority for downstream commercial
    finish/DML stages.  Target LUFS is treated as one input; BAMix handoff,
    reference limiter budget, TP room, input-relative dynamics, artifact risk,
    mono/side risk and optional listener confidence decide whether final push is
    allowed, limited, or replaced by an artifact-protected fallback.
    """
    enabled = bool(_env_bool("BUSY_V6351_MASTERING_OWNERSHIP", True))
    report: dict[str, Any] = {
        "schema_version": "busy_v6351_mastering_ownership_commercial_push_safety_gate",
        "enabled": bool(enabled),
        "active": False,
        "stage": str(stage_label or "finalizer"),
    }
    if not enabled:
        report["reason"] = "disabled_by_env"
        return report

    decision = decision if isinstance(decision, dict) else {}
    cur = current_analysis if isinstance(current_analysis, dict) else {}
    before = before_analysis if isinstance(before_analysis, dict) else {}
    finish = finish_report if isinstance(finish_report, dict) else {}
    hp = _bamix_handoff_policy_from_decision(decision)
    hp = hp if isinstance(hp, dict) else {}
    contract = hp.get("v6341_reference_db_handoff_contract") if isinstance(hp.get("v6341_reference_db_handoff_contract"), dict) else {}
    workload = hp.get("v6350_density_limiter_workload_architecture") if isinstance(hp.get("v6350_density_limiter_workload_architecture"), dict) else {}
    readiness = hp.get("commercial_readiness") if isinstance(hp.get("commercial_readiness"), dict) else {}
    targets = decision.get("targets") if isinstance(decision.get("targets"), dict) else {}

    budget = contract.get("final_limiter_gain_budget_lu") if isinstance(contract.get("final_limiter_gain_budget_lu"), (list, tuple)) else None
    budget_min = _v6351_first_num((budget or [None])[0] if budget else None, default=2.0)
    budget_target = _v6351_first_num((budget or [None, None])[1] if budget and len(budget) > 1 else None, default=3.4)
    budget_max = _v6351_first_num((budget or [None, None, None])[2] if budget and len(budget) > 2 else None, hp.get("max_finish_push_db"), default=4.4)

    target = _v6351_first_num(
        hp.get("effective_final_target_lufs"),
        hp.get("preferred_final_target_lufs"),
        finish.get("target_lufs"),
        targets.get("bamix_mastering_target_lufs"),
        targets.get("integrated_lufs_target"),
        contract.get("delivery_target_lufs"),
        default=None,
    )
    floor = _v6351_first_num(hp.get("effective_floor_lufs"), finish.get("commercial_floor_lufs"), default=(float(target) - 0.75 if target is not None else None))
    premaster_lufs = _v6351_first_num(hp.get("premaster_lufs"), default=None)
    estimated_workload = _v6351_first_num(workload.get("estimated_final_limiter_workload_lu"), default=None)
    if estimated_workload is None and target is not None and premaster_lufs is not None:
        estimated_workload = float(target) - float(premaster_lufs)

    # v63.5.1.1: accept both raw analysis objects and full report objects.
    # Supabase/debug smoke may pass the whole analysis_report, where final
    # metrics live under output_analysis or top-level final_* aliases.
    cur_out = cur.get("output_analysis") if isinstance(cur.get("output_analysis"), dict) else {}
    current_lufs = _v6351_first_num(
        cur.get("integrated_lufs"), cur.get("stereo_integrated_lufs"), cur.get("final_lufs"), cur.get("final_lufs_stereo"),
        cur_out.get("integrated_lufs"), cur_out.get("stereo_integrated_lufs"),
        default=None,
    )
    current_tp = _v6351_first_num(
        cur.get("approx_true_peak_dbfs"), cur.get("true_peak_dbtp"), cur.get("true_peak_dbfs"),
        cur.get("final_true_peak_dbtp"), cur_out.get("approx_true_peak_dbfs"), cur_out.get("true_peak_dbtp"), cur_out.get("true_peak_dbfs"),
        default=None,
    )
    try:
        current_tp = _analysis_true_peak_authority(cur) if current_tp is None else current_tp
    except Exception:
        pass
    current_lra = _analysis_lra_lu(cur) if cur else None
    if current_lra is None and cur_out:
        current_lra = _analysis_lra_lu(cur_out)
    current_crest = _v6351_first_num(cur.get("crest_factor_db"), cur.get("crest_db"), cur.get("final_crest_factor_db"), cur_out.get("crest_factor_db"), cur_out.get("crest_db"), default=None)
    corr = _v6351_first_num(cur.get("correlation"), cur.get("correlation_avg"), cur_out.get("correlation"), cur_out.get("correlation_avg"), default=None)
    side_low = _v6351_first_num(cur.get("side_ratio_lowband"), cur.get("side_low_ratio"), cur_out.get("side_ratio_lowband"), cur_out.get("side_low_ratio"), default=None)
    side_high = _v6351_first_num(cur.get("side_ratio_highband"), cur.get("side_high_ratio"), cur_out.get("side_ratio_highband"), cur_out.get("side_high_ratio"), default=None)

    input_lra = _analysis_lra_lu(before) if before else None
    if input_lra is None:
        input_lra = _v6351_first_num(hp.get("premaster_lra_lu"), default=None)
    input_crest = _v6351_first_num(before.get("crest_factor_db") if before else None, before.get("crest_db") if before else None, hp.get("premaster_crest_factor_db"), default=None)
    lra_delta = (float(current_lra) - float(input_lra)) if current_lra is not None and input_lra is not None else None
    crest_delta = (float(current_crest) - float(input_crest)) if current_crest is not None and input_crest is not None else None
    delivered_gap = max(0.0, float(target) - float(current_lufs)) if target is not None and current_lufs is not None else None
    floor_gap = max(0.0, float(floor) - float(current_lufs)) if floor is not None and current_lufs is not None else None
    excess_over_max = (float(estimated_workload) - float(budget_max)) if estimated_workload is not None and budget_max is not None else _v6351_num(workload.get("workload_excess_over_max_lu"), None)
    excess_over_target = (float(estimated_workload) - float(budget_target)) if estimated_workload is not None and budget_target is not None else _v6351_num(workload.get("workload_excess_over_target_lu"), None)

    mean_art = _v6351_first_num(readiness.get("mean_stem_artifact_risk"), default=0.0)
    max_art = _v6351_first_num(readiness.get("max_stem_artifact_risk"), default=0.0)
    readiness_score = _v6351_first_num(readiness.get("score"), default=None)
    readiness_label = str(readiness.get("label") or "").lower()
    weighted_risk = None
    try:
        gov = decision.get("v62_9_9_research_governance") if isinstance(decision.get("v62_9_9_research_governance"), dict) else {}
        arb = gov.get("final_perceptual_ab_arbiter_policy") if isinstance(gov.get("final_perceptual_ab_arbiter_policy"), dict) else {}
        dmg = arb.get("adaptive_input_relative_damage_scoring") if isinstance(arb.get("adaptive_input_relative_damage_scoring"), dict) else {}
        weighted_risk = _v6351_first_num(dmg.get("weighted_risk"), (dmg.get("decision") or {}).get("weighted_risk") if isinstance(dmg.get("decision"), dict) else None, default=None)
    except Exception:
        weighted_risk = None

    gem = finish.get("final_master_highlight_review") if isinstance(finish.get("final_master_highlight_review"), dict) else {}
    gem_verdict = str(gem.get("verdict") or "").lower()
    gem_commercial = _v6351_first_num(gem.get("commercial_readiness"), default=None)

    profile_text = " ".join([
        str(hp.get("profile") or ""),
        str(hp.get("handoff_mode") or ""),
        str(contract.get("primary_reference_profile_id") or ""),
        _v6351_list_text(contract.get("reference_profile_ids")),
        _v6351_list_text(decision.get("reference_v7") if isinstance(decision.get("reference_v7"), dict) else {}),
        _v6351_list_text(decision.get("genre_chain") if isinstance(decision.get("genre_chain"), dict) else {}),
    ]).lower()
    club_terms = ("club", "edm", "electro", "house", "techno", "dance", "festival", "bounce", "garage", "hardstyle", "schranz")
    streaming_terms = ("streaming_safe", "streaming safe", "safe_intensity", "transparent_polish", "transparent polish", "fragile_protection", "fragile protection")

    hard_stop: list[str] = []
    soft_limit: list[str] = []
    trace: list[dict[str, Any]] = []

    def _trace(code: str, **payload: Any) -> None:
        row = {"code": code}
        for k, v in payload.items():
            if isinstance(v, float):
                row[k] = round(float(v), 4)
            elif v is not None:
                row[k] = v
        trace.append(row)

    if max_art is not None and float(max_art) >= _env_float("BUSY_V6351_MAX_ARTIFACT_HARD", 0.66, 0.2, 1.0):
        hard_stop.append("artifact_risk_high")
    elif mean_art is not None and float(mean_art) >= _env_float("BUSY_V6351_MEAN_ARTIFACT_LIMIT", 0.38, 0.05, 1.0):
        soft_limit.append("artifact_risk_elevated")
    if current_lra is not None and float(current_lra) < _env_float("BUSY_V6351_LRA_HARD_FLOOR_LU", 1.15, 0.2, 4.0):
        hard_stop.append("current_lra_below_artifact_floor")
    elif lra_delta is not None and float(lra_delta) < -_env_float("BUSY_V6351_LRA_LOSS_LIMIT_LU", 2.0, 0.2, 5.0):
        hard_stop.append("input_relative_lra_loss_hard_stop")
    elif lra_delta is not None and float(lra_delta) < -_env_float("BUSY_V6351_LRA_LOSS_SOFT_LU", 1.1, 0.1, 4.0):
        soft_limit.append("input_relative_lra_loss_limit")
    if current_crest is not None and float(current_crest) < _env_float("BUSY_V6351_CREST_HARD_FLOOR_DB", 6.2, 3.0, 12.0):
        hard_stop.append("current_crest_below_artifact_floor")
    elif crest_delta is not None and float(crest_delta) < -_env_float("BUSY_V6351_CREST_LOSS_LIMIT_DB", 3.4, 0.5, 8.0):
        hard_stop.append("input_relative_crest_loss_hard_stop")
    elif crest_delta is not None and float(crest_delta) < -_env_float("BUSY_V6351_CREST_LOSS_SOFT_DB", 2.1, 0.2, 6.0):
        soft_limit.append("input_relative_crest_loss_limit")
    if corr is not None and float(corr) < _env_float("BUSY_V6351_CORRELATION_HARD_FLOOR", 0.24, -1.0, 1.0):
        hard_stop.append("mono_correlation_hard_stop")
    if side_low is not None and float(side_low) > _env_float("BUSY_V6351_LOW_SIDE_HARD_MAX", 0.22, 0.02, 0.8):
        hard_stop.append("low_band_side_translation_hard_stop")
    if weighted_risk is not None and float(weighted_risk) >= _env_float("BUSY_V6351_WEIGHTED_RISK_SOFT_LIMIT", 2.2, 0.0, 5.0):
        soft_limit.append("weighted_damage_risk_limit")
    if excess_over_max is not None and float(excess_over_max) > _env_float("BUSY_V6351_WORKLOAD_EXCESS_SOFT_LU", 0.02, 0.0, 1.0):
        soft_limit.append("reference_limiter_workload_budget_excess")
    if excess_over_max is not None and float(excess_over_max) > _env_float("BUSY_V6351_WORKLOAD_EXCESS_HARD_LU", 0.75, 0.1, 3.0):
        hard_stop.append("reference_limiter_workload_hard_excess")
    if current_tp is not None:
        tp_ceiling_hint = _v6351_first_num(hp.get("true_peak_ceiling_dbtp"), default=float(FINAL_TRUE_PEAK_CEILING_DB))
        tp_room = float(tp_ceiling_hint) - float(current_tp)
        if tp_room < _env_float("BUSY_V6351_TP_PRESSURE_ROOM_DB", 0.08, -0.5, 1.0):
            soft_limit.append("true_peak_room_limited")
    else:
        tp_room = None

    if hard_stop:
        selected = "artifact_protected"
    elif any(term in profile_text for term in streaming_terms) or (target is not None and float(target) <= -11.0):
        selected = "streaming_safe"
    elif any(term in profile_text for term in club_terms) and (target is None or float(target) >= -9.7) and float(max_art or 0.0) <= 0.48:
        selected = "club_forward"
    else:
        selected = "balanced_commercial"

    if selected == "artifact_protected":
        max_limiter_workload = min(float(budget_min or 2.0), float(budget_target or 3.4))
        base_allow = 0.0
        fallback = "streaming_safe"
    elif selected == "streaming_safe":
        max_limiter_workload = float(budget_target or 3.4)
        base_allow = 0.35
        fallback = "artifact_protected" if hard_stop else "streaming_safe"
    elif selected == "club_forward":
        max_limiter_workload = float(budget_max or 4.4)
        base_allow = 1.15
        fallback = "balanced_commercial"
    else:
        max_limiter_workload = min(float(budget_max or 4.4), float(budget_target or 3.4) + 0.45)
        base_allow = 0.75
        fallback = "streaming_safe"

    budget_room_for_extra = None
    if estimated_workload is not None and max_limiter_workload is not None:
        budget_room_for_extra = max(0.0, float(max_limiter_workload) - float(estimated_workload))
    allow_candidates = [float(base_allow)]
    if delivered_gap is not None:
        allow_candidates.append(float(delivered_gap))
    if budget_room_for_extra is not None:
        # A small allowance keeps target-near finals from being blocked by
        # insignificant reference-budget rounding, but prevents rescue-gain use.
        cushion = {"streaming_safe": 0.15, "balanced_commercial": 0.35, "club_forward": 0.55, "artifact_protected": 0.0}.get(selected, 0.25)
        allow_candidates.append(max(0.0, float(budget_room_for_extra) + float(cushion)))
    allowed_push = max(0.0, min(allow_candidates)) if allow_candidates else 0.0
    if soft_limit:
        allowed_push *= _env_float("BUSY_V6351_SOFT_LIMIT_PUSH_SCALE", 0.72, 0.1, 1.0)
    if selected == "artifact_protected":
        allowed_push = 0.0
    allowed_push = float(np.clip(allowed_push, 0.0, 2.0))

    if selected == "artifact_protected":
        authority = "artifact_protected_fallback"
        push_allowed = False
        hard_reason = ";".join(hard_stop) if hard_stop else "artifact_protected_class"
    elif delivered_gap is not None and delivered_gap > allowed_push + 0.05:
        authority = "push_limited"
        push_allowed = bool(allowed_push >= 0.18)
        hard_reason = None
    elif soft_limit:
        authority = "push_limited"
        push_allowed = bool(allowed_push >= 0.18 or (delivered_gap is not None and delivered_gap <= 0.12))
        hard_reason = None
    else:
        authority = "push_allowed"
        push_allowed = True
        hard_reason = None

    tp_policy = {
        "ceiling_dbtp": round(float(_v6351_first_num(hp.get("true_peak_ceiling_dbtp"), default=float(FINAL_TRUE_PEAK_CEILING_DB))), 3),
        "current_true_peak_dbtp": round(float(current_tp), 3) if current_tp is not None else None,
        "room_to_ceiling_db": round(float(tp_room), 3) if tp_room is not None else None,
        "mode": "protected_peak_safety" if selected == "artifact_protected" else ("club_forward_peak_capped" if selected == "club_forward" else "standard_peak_capped"),
        "policy": "TP ceiling is a hard safety boundary; ownership may lower push authority but does not raise the locked ceiling.",
    }

    _trace("inputs", target_lufs=target, floor_lufs=floor, current_lufs=current_lufs, premaster_lufs=premaster_lufs, estimated_workload_lu=estimated_workload)
    _trace("budget", budget_min_lu=budget_min, budget_target_lu=budget_target, budget_max_lu=budget_max, excess_over_max_lu=excess_over_max, max_limiter_workload_lu=max_limiter_workload)
    _trace("damage", input_lra_lu=input_lra, current_lra_lu=current_lra, lra_delta_lu=lra_delta, input_crest_db=input_crest, current_crest_db=current_crest, crest_delta_db=crest_delta)
    _trace("risk", mean_artifact_risk=mean_art, max_artifact_risk=max_art, weighted_risk=weighted_risk, correlation=corr, side_low=side_low, gemini_commercial_readiness=gem_commercial)
    _trace("decision", selected_loudness_class=selected, push_authority=authority, allowed_final_push_db=allowed_push, hard_stop_reason=hard_reason)

    report.update({
        "active": True,
        "selected_loudness_class": selected,
        "push_authority": authority,
        "push_allowed": bool(push_allowed and authority == "push_allowed"),
        "push_limited": bool(authority == "push_limited"),
        "artifact_protected_fallback": bool(authority == "artifact_protected_fallback"),
        "allowed_final_push_db": round(float(allowed_push), 3),
        "final_push_budget_db": round(float(allowed_push), 3),
        "max_limiter_workload_lu": round(float(max_limiter_workload), 3) if max_limiter_workload is not None else None,
        "limiter_workload_budget_lu": round(float(max_limiter_workload), 3) if max_limiter_workload is not None else None,
        "tp_ceiling_policy": tp_policy,
        "fallback_class": fallback,
        "hard_stop_reason": hard_reason,
        "ownership_hard_stop_reason": hard_reason,
        "ownership_fallback_reason": ";".join(hard_stop or soft_limit) if (hard_stop or authority != "push_allowed") else None,
        "soft_limit_reasons": sorted(set(soft_limit)),
        "hard_stop_reasons": sorted(set(hard_stop)),
        "ownership_inputs": {
            "handoff_mode": hp.get("handoff_mode"),
            "profile": hp.get("profile"),
            "reference_primary_profile": contract.get("primary_reference_profile_id"),
            "readiness_label": readiness_label or None,
            "readiness_score": round(float(readiness_score), 3) if readiness_score is not None else None,
            "estimated_final_limiter_workload_lu": round(float(estimated_workload), 3) if estimated_workload is not None else None,
            "reference_limiter_budget_lu": [round(float(budget_min), 3), round(float(budget_target), 3), round(float(budget_max), 3)],
            "workload_excess_over_target_lu": round(float(excess_over_target), 3) if excess_over_target is not None else None,
            "workload_excess_over_max_lu": round(float(excess_over_max), 3) if excess_over_max is not None else None,
            "current_lufs": round(float(current_lufs), 3) if current_lufs is not None else None,
            "current_lra_lu": round(float(current_lra), 3) if current_lra is not None else None,
            "current_crest_db": round(float(current_crest), 3) if current_crest is not None else None,
            "input_lra_lu": round(float(input_lra), 3) if input_lra is not None else None,
            "input_crest_db": round(float(input_crest), 3) if input_crest is not None else None,
            "delivered_gap_to_target_lu": round(float(delivered_gap), 3) if delivered_gap is not None else None,
            "gap_to_floor_lu": round(float(floor_gap), 3) if floor_gap is not None else None,
            "artifact_risk_mean": round(float(mean_art), 4) if mean_art is not None else None,
            "artifact_risk_max": round(float(max_art), 4) if max_art is not None else None,
            "gemini_verdict": gem_verdict or None,
            "gemini_commercial_readiness": round(float(gem_commercial), 3) if gem_commercial is not None else None,
        },
        "ownership_decision_trace": trace,
        "policy": "v63.5.1 classifies loudness ownership before final push: final limiting may polish/catch peaks inside the ownership budget, but target LUFS never overrides artifact, dynamics, TP, translation or BAMix handoff limits.",
    })
    return report


def _v6351_commercial_push_safety_gate_from_ownership(ownership: dict[str, Any] | None) -> dict[str, Any]:
    own = ownership if isinstance(ownership, dict) else {}
    return {
        "schema_version": "busy_v6351_commercial_push_safety_gate",
        "active": bool(own.get("active")),
        "selected_loudness_class": own.get("selected_loudness_class"),
        "push_authority": own.get("push_authority"),
        "push_allowed": bool(own.get("push_allowed")),
        "push_limited": bool(own.get("push_limited")),
        "artifact_protected_fallback": bool(own.get("artifact_protected_fallback")),
        "final_push_budget_db": own.get("final_push_budget_db"),
        "limiter_workload_budget_lu": own.get("limiter_workload_budget_lu"),
        "tp_ceiling_policy": own.get("tp_ceiling_policy"),
        "hard_stop_reason": own.get("hard_stop_reason"),
        "fallback_class": own.get("fallback_class"),
        "mastering_ownership": own,
        "policy": "Compact alias for dashboards: push_allowed/push_limited/artifact_protected_fallback is decided by mastering_ownership.",
    }


def _v6351_select_report_ownership(decision: dict[str, Any] | None = None, finish_report: dict[str, Any] | None = None, dml_report: dict[str, Any] | None = None) -> dict[str, Any]:
    own = _v6351_find_existing_ownership(decision, finish_report, dml_report)
    if own:
        return own
    return {"schema_version": "busy_v6351_mastering_ownership_commercial_push_safety_gate", "enabled": bool(_env_bool("BUSY_V6351_MASTERING_OWNERSHIP", True)), "active": False, "reason": "not_available"}


def _v6351_select_report_gate(decision: dict[str, Any] | None = None, finish_report: dict[str, Any] | None = None, dml_report: dict[str, Any] | None = None) -> dict[str, Any]:
    for src in (finish_report, decision, dml_report):
        if isinstance(src, dict) and isinstance(src.get("commercial_push_safety_gate"), dict):
            return src.get("commercial_push_safety_gate") or {}
        if isinstance(src, dict) and isinstance(src.get("v6350_limiter_workload_budget_plan"), dict):
            gate = (src.get("v6350_limiter_workload_budget_plan") or {}).get("commercial_push_safety_gate")
            if isinstance(gate, dict):
                return gate
    return _v6351_commercial_push_safety_gate_from_ownership(_v6351_select_report_ownership(decision, finish_report, dml_report))


def _v6350_bamix_limiter_workload_budget_plan(
    decision: dict[str, Any] | None,
    standard_final_analysis: dict[str, Any] | None = None,
    finish_report: dict[str, Any] | None = None,
    before_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Surface and enforce the priority-1 final limiter workload contract.

    The finalizer should not silently become a rescue-gain stage.  This helper
    reads the BAMix handoff policy and reference-DB limiter budget, then returns
    conservative DML parameter scales and telemetry for the final report.
    """
    enabled = _env_bool("BUSY_V6350_LIMITER_WORKLOAD_BUDGET", True)
    report: dict[str, Any] = {
        "schema_version": "busy_v6362_bamix_limiter_workload_budget_plan",
        "engine_version": V6362_DML_ENGINE_VERSION,
        "enabled": bool(enabled),
        "active": False,
    }
    if not enabled:
        report["reason"] = "disabled_by_env"
        return report
    hp = _bamix_handoff_policy_from_decision(decision)
    if not isinstance(hp, dict) or not hp.get("active"):
        report["reason"] = "not_bamix_handoff"
        return report
    ownership = _v6351_mastering_ownership_decision(
        decision,
        standard_final_analysis if isinstance(standard_final_analysis, dict) else {},
        finish_report if isinstance(finish_report, dict) else {},
        before_analysis if isinstance(before_analysis, dict) else {},
        stage_label="dml_limiter_workload_budget",
    )
    gate = _v6351_commercial_push_safety_gate_from_ownership(ownership)
    report["mastering_ownership"] = ownership
    report["commercial_push_safety_gate"] = gate
    contract = hp.get("v6341_reference_db_handoff_contract") if isinstance(hp.get("v6341_reference_db_handoff_contract"), dict) else {}
    workload = hp.get("v6350_density_limiter_workload_architecture") if isinstance(hp.get("v6350_density_limiter_workload_architecture"), dict) else {}
    budget = contract.get("final_limiter_gain_budget_lu") if isinstance(contract.get("final_limiter_gain_budget_lu"), (list, tuple)) else None
    try:
        budget_min = float(budget[0]) if budget and len(budget) >= 1 else 2.0
        budget_target = float(budget[1]) if budget and len(budget) >= 2 else 3.4
        budget_max = float(budget[2]) if budget and len(budget) >= 3 else float(hp.get("max_finish_push_db") or 4.4)
    except Exception:
        budget_min, budget_target, budget_max = 2.0, 3.4, float(hp.get("max_finish_push_db") or 4.4)
    def _num(v: Any, default: float | None = None) -> float | None:
        try:
            out = float(v)
            if np.isfinite(out):
                return float(out)
        except Exception:
            pass
        return default
    std = standard_final_analysis if isinstance(standard_final_analysis, dict) else {}
    delivered_lufs = _num(std.get("integrated_lufs"), None)
    delivered_crest = _num(std.get("crest_factor_db"), None)
    delivered_lra = _analysis_lra_lu(std) if isinstance(std, dict) else None
    target = _num(hp.get("effective_final_target_lufs"), None)
    premaster_lufs = _num(hp.get("premaster_lufs"), None)
    premaster_required = float(target - premaster_lufs) if target is not None and premaster_lufs is not None else _num(workload.get("estimated_final_limiter_workload_lu"), None)
    delivered_gap = float(target - delivered_lufs) if target is not None and delivered_lufs is not None else None
    excess_max = float(premaster_required - budget_max) if premaster_required is not None else None
    excess_target = float(premaster_required - budget_target) if premaster_required is not None else None
    fragile = bool((delivered_crest is not None and delivered_crest < _env_float("BUSY_V6350_DML_FRAGILE_CREST_DB", 7.0, 4.0, 12.0)) or (delivered_lra is not None and delivered_lra < _env_float("BUSY_V6350_DML_FRAGILE_LRA_LU", 2.55, 0.5, 7.0)))
    # DML is a final architecture stage. If the premaster already exceeded the
    # reference workload budget, DML may catch peaks but should not increase
    # clipping aggression; upstream density must solve it in the next BAMix pass.
    if fragile:
        clip_scale = _env_float("BUSY_V6350_DML_FRAGILE_CLIP_SCALE", 0.62, 0.2, 1.0)
        slow_scale = _env_float("BUSY_V6350_DML_FRAGILE_SLOW_SCALE", 0.78, 0.2, 1.2)
        role = "peak_safety_fragile_no_density_push"
    elif excess_max is not None and excess_max > _env_float("BUSY_V6350_DML_EXCESS_MAX_TRIGGER_LU", 0.20, 0.0, 2.0):
        clip_scale = _env_float("BUSY_V6350_DML_EXCESS_CLIP_SCALE", 0.82, 0.35, 1.1)
        slow_scale = _env_float("BUSY_V6350_DML_EXCESS_SLOW_SCALE", 0.92, 0.45, 1.2)
        role = "budget_excess_peak_relief_only_upstream_density_required"
    elif delivered_gap is not None and delivered_gap > _env_float("BUSY_V6350_DML_DELIVERED_GAP_TRIGGER_LU", 0.35, 0.0, 2.0):
        clip_scale = _env_float("BUSY_V6350_DML_GAP_CLIP_SCALE", 1.0, 0.5, 1.25)
        slow_scale = _env_float("BUSY_V6350_DML_GAP_SLOW_SCALE", 1.0, 0.5, 1.25)
        role = "bounded_final_density_completion"
    else:
        clip_scale = _env_float("BUSY_V6350_DML_TARGET_MET_CLIP_SCALE", 0.72, 0.25, 1.0)
        slow_scale = _env_float("BUSY_V6350_DML_TARGET_MET_SLOW_SCALE", 0.82, 0.25, 1.1)
        role = "target_met_peak_safety_only"

    # v63.5.1: DML cannot secretly spend more loudness authority than the
    # ownership gate permits.  It may still catch peaks and polish, but if the
    # gate says push_limited/artifact_protected, clipper and slow-limiter
    # aggression are scaled down before the audio path is rendered.
    ownership_authority = str((ownership or {}).get("push_authority") or "") if isinstance(ownership, dict) else ""
    ownership_class = str((ownership or {}).get("selected_loudness_class") or "") if isinstance(ownership, dict) else ""
    if ownership_authority == "artifact_protected_fallback" or ownership_class == "artifact_protected":
        clip_scale = min(float(clip_scale), _env_float("BUSY_V6351_DML_ARTIFACT_CLIP_SCALE", 0.45, 0.1, 1.0))
        slow_scale = min(float(slow_scale), _env_float("BUSY_V6351_DML_ARTIFACT_SLOW_SCALE", 0.62, 0.1, 1.0))
        role = "ownership_artifact_protected_peak_safety_only"
    elif ownership_authority == "push_limited":
        clip_scale = min(float(clip_scale), _env_float("BUSY_V6351_DML_LIMITED_CLIP_SCALE", 0.82, 0.2, 1.1))
        slow_scale = min(float(slow_scale), _env_float("BUSY_V6351_DML_LIMITED_SLOW_SCALE", 0.92, 0.2, 1.1))
        if role == "bounded_final_density_completion":
            role = "ownership_limited_bounded_final_touch"
    report.update({
        "active": True,
        "role": role,
        "handoff_mode": hp.get("handoff_mode"),
        "profile": hp.get("profile"),
        "contract_active": bool(contract.get("active")),
        "final_limiter_budget_lu": [round(float(budget_min), 3), round(float(budget_target), 3), round(float(budget_max), 3)],
        "premaster_lufs": round(float(premaster_lufs), 3) if premaster_lufs is not None else None,
        "effective_final_target_lufs": round(float(target), 3) if target is not None else None,
        "estimated_premaster_to_final_workload_lu": round(float(premaster_required), 3) if premaster_required is not None else None,
        "workload_excess_over_target_lu": round(float(excess_target), 3) if excess_target is not None else None,
        "workload_excess_over_max_lu": round(float(excess_max), 3) if excess_max is not None else None,
        "delivered_lufs_before_dml": round(float(delivered_lufs), 3) if delivered_lufs is not None else None,
        "delivered_gap_to_target_lu": round(float(delivered_gap), 3) if delivered_gap is not None else None,
        "delivered_crest_before_dml": round(float(delivered_crest), 3) if delivered_crest is not None else None,
        "delivered_lra_before_dml": round(float(delivered_lra), 3) if delivered_lra is not None else None,
        "clipper_drive_scale": round(float(clip_scale), 4),
        "slow_limiter_scale": round(float(slow_scale), 4),
        "upstream_workload_router": workload,
        "mastering_ownership": ownership,
        "commercial_push_safety_gate": gate,
        "selected_loudness_class": ownership.get("selected_loudness_class") if isinstance(ownership, dict) else None,
        "push_authority": ownership.get("push_authority") if isinstance(ownership, dict) else None,
        "final_push_budget_db": ownership.get("final_push_budget_db") if isinstance(ownership, dict) else None,
        "ownership_limiter_workload_budget_lu": ownership.get("limiter_workload_budget_lu") if isinstance(ownership, dict) else None,
        "target_miss_allowed_by_ownership": bool((delivered_gap is not None and float(delivered_gap) > 0.05) and (str(ownership.get("push_authority") if isinstance(ownership, dict) else "") in {"push_limited", "artifact_protected_fallback"} or (excess_max is not None and float(excess_max) > 0.0))),
        "dml_contract_route": "mastering_ownership_dml_contract",
        "policy": "Final limiter budget is bounded by BAMix handoff contract and v63.6.2 mastering ownership/DML contract; budget excess is handled upstream or by protected fallback, not by hotter final clipping/limiting.",
    })
    return report

def _v60_deterministic_multistage_limiter(
    source: np.ndarray,
    sr: int,
    before: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    standard_final_analysis: dict[str, Any] | None,
    os_factor: int | None = None,
    finish_report: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Shadow/controlled deterministic multistage limiter candidate.

    This is a foundation layer, not a target-LUFS chaser.  It separates a tiny
    peak-rounding clipper, fast peak limiting, slow macro limiting and TP safety
    so future limiter learning can reason about each stage independently.
    """
    enabled = _env_bool("BUSY_V60_DETERMINISTIC_MULTISTAGE_LIMITER", True)
    apply_enabled = bool(enabled and _env_bool("BUSY_V60_DML_APPLY", True))
    shadow_only = bool(_env_bool("BUSY_V60_DML_SHADOW_ONLY", False))
    report: dict[str, Any] = {
        "schema_version": "busy_v6362_deterministic_multistage_limiter_ownership_contract_stage1",
        "engine_version": V6362_DML_ENGINE_VERSION,
        "enabled": bool(enabled),
        "apply_enabled": bool(apply_enabled),
        "shadow_only": bool(shadow_only),
        "active": False,
        "accepted": False,
        "applied": False,
        "stage_count": 0,
        "policy": "v63.6.2 DML stage-1: HP hygiene -> soft clipper -> fast peak limiter -> slow macro limiter -> true-peak safety, governed by mastering ownership and final-limiter workload contract. Target LUFS is a reference, never a command.",
    }
    src = np.nan_to_num(np.asarray(source, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _duration_sec = float(len(src) / max(1, int(sr))) if getattr(src, "ndim", 1) >= 1 else 0.0
    except Exception:
        _duration_sec = 0.0
    report["duration_sec"] = round(float(_duration_sec), 3)
    if not enabled:
        report["reason"] = "disabled_by_env"
        return src, report
    v6350_limiter_workload_budget = _v6350_bamix_limiter_workload_budget_plan(decision, standard_final_analysis, finish_report, before_analysis=before)
    report["v6350_limiter_workload_budget_plan"] = v6350_limiter_workload_budget
    if isinstance(finish_report, dict):
        finish_report["v6350_limiter_workload_budget_plan"] = v6350_limiter_workload_budget
    if v6350_limiter_workload_budget.get("active"):
        report["schema_version"] = "busy_v6362_deterministic_multistage_limiter_density_workload_contract"
        report["engine_version"] = V6362_DML_ENGINE_VERSION
        report["policy"] = "v63.6.2: deterministic multistage limiter remains bounded by BAMix handoff workload and mastering ownership contract; density must be created upstream, not by overdriving the final limiter."
    # v8.5.3.62.8.2: DML is an experimental candidate foundation layer.  On
    # full-length Cloud Run jobs its multiple internal limiter/analyzer passes can
    # dominate wall time even though the outer final limiter/export lock remains
    # authoritative.  Keep production runtime safe by bypassing DML on longer
    # tracks unless it is explicitly forced.
    try:
        _runtime_safe_bypass = _env_bool("BUSY_V60_DML_RUNTIME_SAFE_BYPASS", False)
        _runtime_bypass_min_sec = _env_float("BUSY_V60_DML_RUNTIME_BYPASS_MIN_SEC", 12.0, 3.0, 600.0)
        _runtime_force_full = _env_bool("BUSY_V60_DML_FORCE_FULL_LENGTH", False)
        if _env_bool("BUSY_V6316_RECONNECT_V60_FULL_LENGTH", True):
            _runtime_safe_bypass = False
            _runtime_force_full = True
    except Exception:
        _runtime_safe_bypass, _runtime_bypass_min_sec, _runtime_force_full = True, 12.0, False
    if bool(_runtime_safe_bypass) and not bool(_runtime_force_full) and float(_duration_sec) >= float(_runtime_bypass_min_sec):
        report.update({
            "active": False,
            "accepted": False,
            "applied": False,
            "reason": "runtime_safe_bypass_full_length_track",
            "runtime_safe_bypass": True,
            "runtime_bypass_min_sec": round(float(_runtime_bypass_min_sec), 3),
            "policy_runtime": "Full-length DML is bypassed by default for Cloud Run wall-time safety; final limiter/export-lock analysis remains authoritative. Set BUSY_V60_DML_FORCE_FULL_LENGTH=1 to force DML.",
        })
        return src, report
    try:
        os_use = int(np.clip(int(os_factor or _qc_oversample_factor()), 2, 32))
        # v8.5.3.62.8.2: DML is a candidate foundation stage, not the final
        # authoritative true-peak/export measurement pass.  Running its internal
        # two limiter stages at the full final oversample factor can dominate wall
        # time on Cloud Run.  Keep DML bounded by default while the outer final
        # limiter/export lock still owns the authoritative ceiling check.
        try:
            _dml_os_cap = int(_env_float("BUSY_V60_DML_OS_FACTOR_MAX", 2.0, 2.0, 8.0))
            os_use = int(min(os_use, max(2, _dml_os_cap)))
        except Exception:
            os_use = min(os_use, 2)
    except Exception:
        os_use = int(_qc_oversample_factor())
    try:
        limiter_cfg = decision.get("limiter", {}) if isinstance(decision, dict) else {}
        legacy_lookahead = limiter_cfg.get("lookahead_ms", 1.0) if isinstance(limiter_cfg, dict) else 1.0
        legacy_release = limiter_cfg.get("release_ms", 160.0) if isinstance(limiter_cfg, dict) else 160.0
        if isinstance(legacy_lookahead, list):
            legacy_lookahead = float(sum(legacy_lookahead) / max(1, len(legacy_lookahead)))
        if isinstance(legacy_release, list):
            legacy_release = float(sum(legacy_release) / max(1, len(legacy_release)))
        clip_drive_base = _env_float("BUSY_V60_DML_SOFT_CLIP_DRIVE_DB", 0.35, 0.0, 1.5)
        clip_mix_base = _env_float("BUSY_V60_DML_SOFT_CLIP_MIX", 0.16, 0.0, 0.55)
        fast_lookahead = _env_float("BUSY_V60_DML_FAST_LOOKAHEAD_MS", 0.55, 0.20, 2.0)
        fast_release = _env_float("BUSY_V60_DML_FAST_RELEASE_MS", 55.0, 25.0, 130.0)
        slow_lookahead = _env_float("BUSY_V60_DML_SLOW_LOOKAHEAD_MS", max(1.0, float(legacy_lookahead)), 0.45, 5.0)
        slow_release_base = _env_float("BUSY_V60_DML_SLOW_RELEASE_MS", max(140.0, float(legacy_release)), 80.0, 280.0)
        clip_scale = 1.0
        slow_scale = 1.0
        if isinstance(v6350_limiter_workload_budget, dict) and v6350_limiter_workload_budget.get("active"):
            try:
                clip_scale = float(v6350_limiter_workload_budget.get("clipper_drive_scale", 1.0) or 1.0)
                slow_scale = float(v6350_limiter_workload_budget.get("slow_limiter_scale", 1.0) or 1.0)
            except Exception:
                clip_scale, slow_scale = 1.0, 1.0
        clip_drive = float(clip_drive_base) * float(np.clip(clip_scale, 0.2, 1.25))
        clip_mix = float(clip_mix_base) * float(np.clip(0.70 + 0.30 * clip_scale, 0.45, 1.10))
        # Reduce macro-stage pressure when the handoff contract says the limiter
        # should only catch peaks.  This avoids turning DML into a hidden gain
        # rescue path when upstream density was insufficient.
        slow_release = float(slow_release_base) * float(np.clip(1.0 + max(0.0, 1.0 - slow_scale) * 0.35, 0.85, 1.35))
        fast_ceiling = min(float(FINAL_TRUE_PEAK_CEILING_DB) + 0.45, -0.35)
        slow_margin = 0.12 * float(np.clip(slow_scale, 0.35, 1.20))
        slow_ceiling = min(float(FINAL_TRUE_PEAK_CEILING_DB) + slow_margin, -0.55)
        stages: list[dict[str, Any]] = []
        def snap(name: str, audio: np.ndarray) -> dict[str, Any]:
            an = analyze_audio_fast_qc(audio, sr, true_peak_oversample=_qc_oversample_factor())
            an = _v6362_ensure_lra_for_dml(an, audio, sr, source=name)
            row = {"stage": name, "analysis": _analysis_summary_for_loudness(an)}
            stages.append(row)
            return an
        x = src
        snap("input", x)
        # Stage 0: only removes subsonic/DC material; no tonal boost.
        try:
            x0 = highpass(x, sr, cutoff_hz=_env_float("BUSY_V60_DML_HPF_HZ", 18.0, 8.0, 32.0), order=2)
        except Exception:
            x0 = x
        snap("stage0_hpf_hygiene", x0)
        # Stage 1: tiny soft-knee rounding before any limiter.  This is not a loudness drive.
        x1 = soft_clip(x0, drive_db=float(clip_drive), mix=float(clip_mix))
        snap("stage1_soft_clip_peak_rounding", x1)
        # Stage 2: fast peak catcher, intentionally above the final TP ceiling.
        x2 = oversampled_limit_chunked(x1, sr, ceiling_db=float(fast_ceiling), oversample=int(os_use), lookahead_ms=float(fast_lookahead), release_ms=float(fast_release))
        snap("stage2_fast_peak_limiter", x2)
        # Stage 3: slower macro limiter to avoid making the TP stage do all work.
        x3 = oversampled_limit_chunked(x2, sr, ceiling_db=float(slow_ceiling), oversample=int(os_use), lookahead_ms=float(slow_lookahead), release_ms=float(slow_release))
        snap("stage3_slow_macro_limiter", x3)
        # Stage 4: true-peak safety only.
        x4, tp_gain = _normalize_to_true_peak(x3, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=int(os_use))
        post = snap("stage4_true_peak_safety", x4)
        post_full: dict[str, Any] = {}
        post_full_error: str | None = None
        if _env_bool("BUSY_V60_DML_FULL_ANALYSIS", False):
            try:
                post_full = analyze_audio(x4, sr)
            except Exception as exc:
                post_full_error = str(exc)[:240]
        comparison_post = post_full if post_full else post
        acceptance = _v60_candidate_acceptance(before, standard_final_analysis, comparison_post, decision, finish_report=finish_report)
        applied = bool(apply_enabled and not shadow_only and acceptance.get("accepted"))
        report.update({
            "active": True,
            "stage_count": 5,
            "oversample_factor": int(os_use),
            "parameters": {
                "soft_clip_drive_db": round(float(clip_drive), 3),
                "soft_clip_mix": round(float(clip_mix), 3),
                "v6350_soft_clip_drive_base_db": round(float(clip_drive_base), 3),
                "v6350_soft_clip_mix_base": round(float(clip_mix_base), 3),
                "v6350_clipper_drive_scale": round(float(clip_scale), 4),
                "v6350_slow_limiter_scale": round(float(slow_scale), 4),
                "fast_ceiling_dbtp": round(float(fast_ceiling), 3),
                "fast_lookahead_ms": round(float(fast_lookahead), 3),
                "fast_release_ms": round(float(fast_release), 3),
                "slow_ceiling_dbtp": round(float(slow_ceiling), 3),
                "slow_lookahead_ms": round(float(slow_lookahead), 3),
                "slow_release_ms": round(float(slow_release), 3),
                "true_peak_safety_gain_db": round(float(tp_gain), 3),
            },
            "stages": stages,
            "post_analysis": _analysis_summary_for_loudness(post),
            "post_analysis_full": _analysis_summary_for_loudness(post_full) if post_full else {},
            "post_analysis_full_error": post_full_error,
            "standard_final_analysis": _analysis_summary_for_loudness(standard_final_analysis if isinstance(standard_final_analysis, dict) else {}),
            "comparison_analysis_mode": "full_analyze_audio" if post_full else "fast_qc_fallback",
            "mastering_ownership": (v6350_limiter_workload_budget.get("mastering_ownership") if isinstance(v6350_limiter_workload_budget, dict) else {}) or {},
            "commercial_push_safety_gate": (v6350_limiter_workload_budget.get("commercial_push_safety_gate") if isinstance(v6350_limiter_workload_budget, dict) else {}) or {},
            "v6362_dml_ownership_contract": acceptance.get("v6362_dml_ownership_contract") if isinstance(acceptance.get("v6362_dml_ownership_contract"), dict) else {},
            "acceptance_gate": acceptance,
            "accepted": bool(acceptance.get("accepted")),
            "applied": bool(applied),
            "reason": "accepted_and_applied" if applied else ("accepted_shadow_only" if acceptance.get("accepted") else "candidate_rejected_by_damage_gate"),
        })
        return (np.nan_to_num(x4, nan=0.0, posinf=0.0, neginf=0.0) if applied else src), report
    except Exception as exc:
        report.update({"active": False, "accepted": False, "applied": False, "reason": "exception", "error": str(exc)[:300]})
        return src, report



def _v63162_post_safety_limit_candidate(
    audio: np.ndarray,
    sr: int,
    limiter_cfg: dict[str, Any] | None = None,
    label: str = "candidate",
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Apply the same final TP safety context used by delivered masters.

    v63.1.6.2 uses this for proxy/stress candidates before scoring.  The
    important distinction is candidate -> final safety limiter -> post-QC.  A
    pre-safety TP spike is telemetry, not an automatic reject.
    """
    cfg = limiter_cfg if isinstance(limiter_cfg, dict) else {}
    try:
        lookahead_ms = cfg.get("lookahead_ms", 1.0)
        release_ms = cfg.get("release_ms", 160.0)
        if isinstance(lookahead_ms, list):
            lookahead_ms = float(sum(lookahead_ms) / max(1, len(lookahead_ms)))
        if isinstance(release_ms, list):
            release_ms = float(sum(release_ms) / max(1, len(release_ms)))
    except Exception:
        lookahead_ms, release_ms = 1.0, 160.0
    report: dict[str, Any] = {
        "schema_version": "busy_v63162_post_safety_candidate_qc",
        "active": True,
        "label": str(label),
        "target_true_peak_dbtp": round(float(FINAL_TRUE_PEAK_CEILING_DB), 3),
        "policy": "Candidate QC is performed after final safety limiting/TP attenuation; pre-safety true peak is not a hard reject by itself.",
    }
    x = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        pre = analyze_audio_fast_qc(x, sr, true_peak_oversample=_qc_oversample_factor())
        report["pre_safety_analysis"] = _analysis_summary_for_loudness(pre)
    except Exception as exc:
        report["pre_safety_analysis_error"] = str(exc)[:220]
    try:
        y = oversampled_limit_chunked(
            x,
            sr,
            ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
            oversample=_qc_oversample_factor(),
            lookahead_ms=float(lookahead_ms),
            release_ms=float(release_ms),
        )
    except Exception:
        y = x
        report["limiter_fallback"] = "oversampled_limit_chunked_exception_raw_then_tp_attenuation"
    try:
        y, gain_db, att = _attenuate_to_true_peak_ceiling(y, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=_qc_oversample_factor())
        report["true_peak_attenuation"] = att
        report["true_peak_attenuation_gain_db"] = round(float(gain_db), 3)
    except Exception as exc:
        report["true_peak_attenuation_error"] = str(exc)[:220]
    post = analyze_audio_fast_qc(y, sr, true_peak_oversample=_qc_oversample_factor())
    report["post_safety_analysis"] = _analysis_summary_for_loudness(post)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False), post, report


def _v63162_commercial_floor_recovery_candidate(
    audio: np.ndarray,
    sr: int,
    before: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    mode: str,
    standard_analysis: dict[str, Any] | None,
    limiter_cfg: dict[str, Any] | None = None,
    label: str = "commercial_floor_recovery",
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Try one bounded post-safety lift when commercial finish is safe but below floor."""
    profile = _commercial_loudness_finish_profile(decision or {}, mode)
    floor_lufs = _v60_float_metric(profile, "commercial_floor_lufs", default=_v60_float_metric(profile, "target_lufs", default=-12.0))
    target_lufs = _v60_float_metric(profile, "target_lufs", default=floor_lufs)
    recovery_margin = _env_float("BUSY_V63162_COMMERCIAL_FLOOR_RECOVERY_MARGIN_LU", 0.06, 0.0, 0.35)
    tol = _env_float("BUSY_V63162_COMMERCIAL_FLOOR_TOLERANCE_LU", 0.04, 0.0, 0.20)
    max_lift = _env_float("BUSY_V63162_COMMERCIAL_FLOOR_RECOVERY_MAX_LIFT_DB", 1.60, 0.10, 3.0)
    report: dict[str, Any] = {
        "schema_version": "busy_v63162_commercial_floor_recovery_candidate",
        "active": False,
        "accepted": False,
        "applied": False,
        "label": str(label),
        "commercial_floor_lufs": round(float(floor_lufs), 3) if floor_lufs is not None else None,
        "target_lufs": round(float(target_lufs), 3) if target_lufs is not None else None,
        "policy": "A safe checkpoint below the commercial floor is partial evidence, not final selected output. One bounded post-safety lift is evaluated against final QC before fallback to standard final.",
    }
    x = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    try:
        pre = analyze_audio_fast_qc(x, sr, true_peak_oversample=_qc_oversample_factor())
        report["input_analysis"] = _analysis_summary_for_loudness(pre)
        current = float(pre.get("integrated_lufs", -99.0) or -99.0)
        if floor_lufs is None:
            report["reason"] = "missing_commercial_floor"
            return x, pre, report
        needed = float(floor_lufs) - current + float(recovery_margin)
        report["needed_lift_db"] = round(float(needed), 3)
        if needed <= -float(tol):
            safe, post, safety = _v63162_post_safety_limit_candidate(x, sr, limiter_cfg=limiter_cfg, label=str(label) + "_already_floor_met")
            report.update({"active": True, "accepted": True, "applied": True, "reason": "already_floor_met", "post_safety_qc": safety, "output_analysis": _analysis_summary_for_loudness(post)})
            return safe, post, report
        if needed > max_lift:
            report["reason"] = "needed_lift_exceeds_recovery_cap"
            return x, pre, report
        recovery_target = min(float(target_lufs) if target_lufs is not None else float(floor_lufs) + recovery_margin, float(floor_lufs) + max(0.02, recovery_margin))
        report["recovery_target_lufs"] = round(float(recovery_target), 3)
        lifted, iter_report = _loudness_iteration(x, sr, target_lufs=float(recovery_target), ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, limiter=limiter_cfg or {}, max_iters=5)
        report["loudness_iteration"] = iter_report
        safe, post, safety = _v63162_post_safety_limit_candidate(lifted, sr, limiter_cfg=limiter_cfg, label=str(label) + "_post_lift")
        post_lufs = _v60_float_metric(post, "integrated_lufs", default=-99.0)
        post_tp = _v60_float_metric(post, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
        std = standard_analysis if isinstance(standard_analysis, dict) else {}
        cand_risk = _safety_risk(before or {}, post, decision or {}) if isinstance(post, dict) else []
        std_risk = _safety_risk(before or {}, std, decision or {}) if isinstance(std, dict) and std else []
        hard = {"phase_collapse", "low_end_collapse", "vocal_distortion", "clipping_distortion_risk"}
        new_hard = [r for r in cand_risk if r not in set(std_risk) and r in hard]
        lra_std = _v60_float_metric(std, "lra_lu", "loudness_range_lu", default=None)
        lra_cand = _v60_float_metric(post, "lra_lu", "loudness_range_lu", default=None)
        crest_std = _v60_float_metric(std, "crest_factor_db", default=None)
        crest_cand = _v60_float_metric(post, "crest_factor_db", default=None)
        reasons: list[str] = []
        if post_lufs is None or float(post_lufs) < float(floor_lufs) - float(tol):
            reasons.append("floor_still_not_reached_after_recovery")
        if post_tp is not None and float(post_tp) > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.04:
            reasons.append("post_recovery_true_peak_over_ceiling")
        if lra_std is not None and lra_cand is not None and float(lra_cand) < float(lra_std) - _env_float("BUSY_V63162_FLOOR_RECOVERY_MAX_LRA_LOSS_LU", 0.25, 0.0, 1.0):
            reasons.append("post_recovery_lra_loss_exceeds_gate")
        if crest_std is not None and crest_cand is not None and float(crest_cand) < float(crest_std) - _env_float("BUSY_V63162_FLOOR_RECOVERY_MAX_CREST_LOSS_DB", 0.45, 0.0, 2.0):
            reasons.append("post_recovery_crest_loss_exceeds_gate")
        if new_hard:
            reasons.append("post_recovery_new_hard_safety_risk")
        accepted = bool(not reasons)
        report.update({
            "active": True,
            "accepted": accepted,
            "applied": accepted,
            "reason": "accepted_post_safety_floor_recovery" if accepted else "rejected_post_safety_floor_recovery",
            "reject_reasons": reasons,
            "post_safety_qc": safety,
            "output_analysis": _analysis_summary_for_loudness(post),
            "candidate_safety_risk": cand_risk,
            "standard_safety_risk": std_risk,
        })
        return (safe if accepted else x), (post if accepted else pre), report
    except Exception as exc:
        report.update({"active": False, "accepted": False, "applied": False, "reason": "exception", "error": str(exc)[:260]})
        try:
            pre = analyze_audio_fast_qc(x, sr, true_peak_oversample=_qc_oversample_factor())
        except Exception:
            pre = {}
        return x, pre, report

def _final_grade_selected_candidate_render(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    limiter_cfg: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the final-grade limiter/TP-normalizer to the selected candidate once.

    Commercial candidate renders use the cheap QC oversampling path so several
    candidates can be evaluated without exploding wall time.  Once a commercial
    candidate is selected, the delivered WAV must not be the cheap candidate
    render.  This final-grade pass uses the same adaptive BUSY_OVERSAMPLE policy
    as the standard final limiter and records both the candidate-render and final
    true-peak precision explicitly in stage_state.
    """
    enabled = _env_bool("BUSY_V48_2_1_FINAL_GRADE_SELECTED_CANDIDATE", True)
    report: dict[str, Any] = {
        "schema_version": "busy_selected_candidate_final_grade_render_v8_5_3_48_2_1",
        "enabled": bool(enabled),
        "active": False,
        "candidate_render_oversample_factor": int(_qc_oversample_factor()),
        "policy": "full-render candidates may use cheap QC oversampling; the selected candidate is passed once through final-grade limiter/TP normalization before export.",
    }
    if not enabled:
        report["reason"] = "disabled_by_env"
        return np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), report
    try:
        os_factor, adaptive_quality = _effective_oversample_factor(before or {}, decision or {})
        try:
            _final_grade_os_cap = int(_env_float("BUSY_SELECTED_CANDIDATE_FINAL_GRADE_OS_FACTOR_MAX", 2.0, 2.0, 8.0))
            os_factor = int(min(int(os_factor), max(2, _final_grade_os_cap)))
            adaptive_quality = {**(adaptive_quality if isinstance(adaptive_quality, dict) else {}), "selected_candidate_runtime_os_cap": int(os_factor), "runtime_cap_policy": "v62.8.2 caps selected-candidate final-grade render; returned-buffer final analysis/export lock remain authoritative"}
        except Exception:
            os_factor = min(int(os_factor), 2)
        cfg = limiter_cfg if isinstance(limiter_cfg, dict) else {}
        lookahead_ms = cfg.get("lookahead_ms", 1.0)
        release_ms = cfg.get("release_ms", 160.0)
        if isinstance(lookahead_ms, list):
            lookahead_ms = float(sum(lookahead_ms) / max(1, len(lookahead_ms)))
        if isinstance(release_ms, list):
            release_ms = float(sum(release_ms) / max(1, len(release_ms)))
        _fg_started_at = time.monotonic()
        try:
            _fg_duration_sec = float(len(y) / max(1, int(sr)))
        except Exception:
            _fg_duration_sec = 0.0
        _cached_candidate_analysis = cfg.get("candidate_analysis") if isinstance(cfg.get("candidate_analysis"), dict) else {}
        if _env_bool("BUSY_SELECTED_CANDIDATE_REUSE_LONG_HOT_RENDER", True) and _fg_duration_sec >= _env_float("BUSY_SELECTED_CANDIDATE_REUSE_LONG_MIN_SEC", 12.0, 3.0, 600.0) and _cached_candidate_analysis:
            pre = _cached_candidate_analysis
            rendered = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            buffer_identity = _v56_audio_buffer_identity(
                rendered,
                sr,
                pre,
                stage="selected_candidate_hot_render_reused_long_track_runtime_safe",
                source="commercial_candidate",
                candidate_label=str((limiter_cfg or {}).get("candidate_label") or "") or None,
            )
            report.update({
                "active": True,
                "schema_version": "busy_selected_candidate_final_grade_render_v8_5_3_62_8_2_runtime_hotfix",
                "final_grade_oversample_factor": int(os_factor),
                "adaptive_quality": adaptive_quality,
                "pre_analysis": _analysis_summary_for_loudness(pre),
                "post_analysis": _analysis_summary_for_loudness(pre),
                "post_analysis_full": {},
                "post_analysis_full_enabled": False,
                "buffer_identity": buffer_identity,
                "runtime_short_circuit": True,
                "runtime_long_track_short_circuit": True,
                "reason": "long_track_hot_render_candidate_reused_with_cached_analysis_for_runtime_safety",
                "policy": "selected hot candidate already passed candidate limiter/TP normalization; long-track final-grade duplicate limiter/analyzer pass is skipped for Cloud Run wall-time safety",
            })
            _debug("selected_candidate_final_grade_reused_hot_render", elapsed_sec=round(float(time.monotonic() - _fg_started_at), 3), true_peak_db=(pre.get("approx_true_peak_dbfs") if isinstance(pre, dict) else None), long_track=True)
            return rendered, report
        pre = analyze_audio_fast_qc(y, sr, true_peak_oversample=_qc_oversample_factor())
        try:
            _pre_tp = float(pre.get("approx_true_peak_dbfs", pre.get("true_peak_dbfs", -99.0)) or -99.0)
        except Exception:
            _pre_tp = -99.0
        if _env_bool("BUSY_SELECTED_CANDIDATE_TRUST_HOT_RENDER_TP_SAFE", True) and _pre_tp <= float(FINAL_TRUE_PEAK_CEILING_DB) + 0.03:
            rendered = np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            buffer_identity = _v56_audio_buffer_identity(
                rendered,
                sr,
                pre,
                stage="selected_candidate_hot_render_reused",
                source="commercial_candidate",
                candidate_label=str((limiter_cfg or {}).get("candidate_label") or "") or None,
            )
            report.update({
                "active": True,
                "schema_version": "busy_selected_candidate_final_grade_render_v8_5_3_62_8_2_runtime_hotfix",
                "final_grade_oversample_factor": int(os_factor),
                "adaptive_quality": adaptive_quality,
                "pre_analysis": _analysis_summary_for_loudness(pre),
                "post_analysis": _analysis_summary_for_loudness(pre),
                "post_analysis_full": {},
                "post_analysis_full_enabled": False,
                "buffer_identity": buffer_identity,
                "runtime_short_circuit": True,
                "reason": "hot_render_candidate_already_true_peak_safe_reused_for_runtime_safety",
                "policy": "selected hot candidate already passed candidate limiter/TP normalization; final returned-buffer analysis/export lock remains authoritative",
            })
            _debug("selected_candidate_final_grade_reused_hot_render", elapsed_sec=round(float(time.monotonic() - _fg_started_at), 3), true_peak_db=round(float(_pre_tp), 3))
            return rendered, report
        _debug("selected_candidate_final_grade_start", shape=getattr(y, "shape", None), os_factor=int(os_factor))
        rendered = oversampled_limit_chunked(
            np.asarray(y, dtype=np.float32),
            sr,
            ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
            oversample=int(os_factor),
            lookahead_ms=float(lookahead_ms),
            release_ms=float(release_ms),
        )
        rendered, tp_gain = _normalize_to_true_peak(rendered, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=int(os_factor))
        post = analyze_audio_fast_qc(rendered, sr, true_peak_oversample=int(os_factor))
        _debug("selected_candidate_final_grade_done", elapsed_sec=round(float(time.monotonic() - _fg_started_at), 3), os_factor=int(os_factor))
        post_full_summary: dict[str, Any] = {}
        full_analysis_error: str | None = None
        if _env_bool("BUSY_V56_SELECTED_CANDIDATE_FULL_EXPORT_ANALYSIS", False):
            try:
                post_full = analyze_audio(rendered, sr)
                post_full_summary = _analysis_summary_for_loudness(post_full)
            except Exception as exc:
                full_analysis_error = str(exc)[:240]
        buffer_identity = _v56_audio_buffer_identity(
            rendered,
            sr,
            post_full_summary or post,
            stage="selected_candidate_final_grade_render",
            source="commercial_candidate",
            candidate_label=str((limiter_cfg or {}).get("candidate_label") or "") or None,
        )
        report.update({
            "active": True,
            "schema_version": "busy_selected_candidate_final_grade_render_v8_5_3_56_export_identity_lock",
            "final_grade_oversample_factor": int(os_factor),
            "adaptive_quality": adaptive_quality,
            "lookahead_ms": round(float(lookahead_ms), 3),
            "release_ms": round(float(release_ms), 3),
            "true_peak_normalize_gain_db": round(float(tp_gain), 3),
            "pre_analysis": _analysis_summary_for_loudness(pre),
            "post_analysis": _analysis_summary_for_loudness(post),
            "post_analysis_full": post_full_summary,
            "post_analysis_full_enabled": _env_bool("BUSY_V56_SELECTED_CANDIDATE_FULL_EXPORT_ANALYSIS", False),
            "post_analysis_full_error": full_analysis_error,
            "buffer_identity": buffer_identity,
            "reason": "selected_candidate_promoted_to_final_grade_render_with_export_identity",
        })
        return np.nan_to_num(rendered, nan=0.0, posinf=0.0, neginf=0.0), report
    except Exception as exc:
        report.update({
            "active": False,
            "reason": "exception_preserve_selected_candidate",
            "error": str(exc)[:300],
        })
        return np.nan_to_num(np.asarray(y, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0), report


def _pre_clean_makeup_compensation(
    y: np.ndarray,
    sr: int,
    original_analysis: dict[str, Any],
    cleaned_analysis: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Restore accidental loudness loss from residue pre-clean while preserving headroom."""
    enabled = _env_bool("BUSY_PRE_CLEAN_MAKEUP_GAIN", True)
    report: dict[str, Any] = {
        "enabled": enabled,
        "schema_version": "busy_pre_clean_makeup_v8_5_3_20",
        "policy": "compensate cleanup-induced loudness loss before premaster only when true-peak headroom allows it",
        "applied": False,
    }
    if not enabled:
        report["reason"] = "disabled"
        return y, report, cleaned_analysis
    try:
        orig_lufs = float(original_analysis.get("integrated_lufs", -99.0) or -99.0)
        clean_lufs = float(cleaned_analysis.get("integrated_lufs", -99.0) or -99.0)
        clean_tp = float(cleaned_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0)
        if not all(np.isfinite(v) for v in (orig_lufs, clean_lufs, clean_tp)):
            report["reason"] = "non_finite_metrics"
            return y, report, cleaned_analysis
        loss_db = orig_lufs - clean_lufs
        min_loss = _env_float("BUSY_PRE_CLEAN_MAKEUP_MIN_LOSS_DB", 0.45, 0.05, 3.0)
        max_makeup = _env_float("BUSY_PRE_CLEAN_MAKEUP_MAX_DB", 1.75, 0.0, 4.0)
        target_tp = _env_float("BUSY_PRE_CLEAN_MAKEUP_TARGET_TP_DB", -1.05, -4.0, -0.25)
        headroom_gain = target_tp - clean_tp
        makeup_db = float(max(0.0, min(loss_db, max_makeup, headroom_gain)))
        report.update({
            "original_lufs": round(orig_lufs, 3),
            "cleaned_lufs_before_makeup": round(clean_lufs, 3),
            "cleaned_true_peak_before_makeup_db": round(clean_tp, 3),
            "cleanup_lufs_loss_db": round(loss_db, 3),
            "target_true_peak_db": round(target_tp, 3),
            "headroom_limited_gain_db": round(max(0.0, headroom_gain), 3),
            "max_makeup_db": round(max_makeup, 3),
        })
        if loss_db < min_loss:
            report["reason"] = "cleanup_lufs_loss_below_threshold"
            return y, report, cleaned_analysis
        if makeup_db <= 0.02:
            report["reason"] = "no_true_peak_headroom_for_makeup"
            return y, report, cleaned_analysis
        out = apply_gain_db(y, makeup_db)
        updated = analyze_audio_fast_qc(out, sr, true_peak_oversample=_qc_oversample_factor())
        report.update({
            "applied": True,
            "makeup_gain_db": round(makeup_db, 3),
            "cleaned_lufs_after_makeup": round(float(updated.get("integrated_lufs", clean_lufs) or clean_lufs), 3),
            "cleaned_true_peak_after_makeup_db": round(float(updated.get("approx_true_peak_dbfs", clean_tp) or clean_tp), 3),
            "reason": "pre_clean_loudness_loss_compensated_with_headroom_guard",
        })
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report, updated
    except Exception as exc:
        report.update({"reason": "error", "error_type": type(exc).__name__, "error": str(exc)[:300]})
        return y, report, cleaned_analysis


def _commercial_hot_candidate_render_with_trace(
    x_after_gain: np.ndarray,
    sr: int,
    ceiling_db: float,
    target_lufs: float,
    floor_lufs: float,
    before: dict[str, Any],
    decision: dict[str, Any],
    profile: dict[str, Any],
    render_controls: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a hot-commercial candidate without letting the limiter erase the gain.

    If a high-gain candidate goes straight into a transparent limiter, peak
    overshoot can make the limiter turn the whole song down.  This path records
    every loudness stage and adds peak absorption/density recovery so the virtual
    strategy is actually translated into final LUFS.
    """
    trace: list[dict[str, Any]] = []
    os_factor = _qc_oversample_factor()
    render_controls = render_controls if isinstance(render_controls, dict) else {}
    peak_absorb_drive_scale = float(np.clip(float(render_controls.get("peak_absorb_drive_scale", 1.0) or 1.0), 0.35, 1.15))
    peak_absorb_mix_scale = float(np.clip(float(render_controls.get("peak_absorb_mix_scale", 1.0) or 1.0), 0.30, 1.15))
    density_recovery_allowed = bool(render_controls.get("density_recovery_allowed", True))
    density_recovery_gain_scale = float(np.clip(float(render_controls.get("density_recovery_gain_scale", 1.0) or 1.0), 0.20, 1.10))
    density_recovery_drive_scale = float(np.clip(float(render_controls.get("density_recovery_drive_scale", 1.0) or 1.0), 0.20, 1.10))
    density_recovery_mix_scale = float(np.clip(float(render_controls.get("density_recovery_mix_scale", 1.0) or 1.0), 0.20, 1.10))
    density_recovery_max_passes = int(np.clip(int(render_controls.get("density_recovery_max_passes", 4) or 4), 0, 4))
    if _env_bool("BUSY_COMMERCIAL_FINISH_FAST_RENDER", True):
        density_recovery_max_passes = min(density_recovery_max_passes, int(_env_float("BUSY_COMMERCIAL_FAST_DENSITY_RECOVERY_MAX_PASSES", 1, 0, 3)))
    limiter_release_ms = float(np.clip(float(render_controls.get("limiter_release_ms", 115.0) or 115.0), 70.0, 198.0))
    limiter_lookahead_ms = float(np.clip(float(render_controls.get("limiter_lookahead_ms", 0.8) or 0.8), 0.45, 1.55))
    density_recovery_limiter_release_ms = float(np.clip(float(render_controls.get("density_recovery_limiter_release_ms", 90.0) or 90.0), 70.0, 170.0))
    density_recovery_limiter_lookahead_ms = float(np.clip(float(render_controls.get("density_recovery_limiter_lookahead_ms", 0.65) or 0.65), 0.45, 1.35))
    v55_density_lift_scale = float(np.clip(float(render_controls.get("v55_density_lift_scale", 1.0) or 1.0), 0.80, 1.20))

    def snap(label: str, audio: np.ndarray) -> dict[str, Any]:
        an = analyze_audio_fast_qc(audio, sr, true_peak_oversample=os_factor)
        item = {"stage": label, **_analysis_summary_for_loudness(an)}
        trace.append(item)
        return an

    out = np.nan_to_num(np.asarray(x_after_gain, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    start_an = snap("after_strategy_input_gain", out)
    start_lufs = float(start_an.get("integrated_lufs", -99.0) or -99.0)
    start_tp = float(start_an.get("approx_true_peak_dbfs", -99.0) or -99.0)
    start_crest = float(start_an.get("crest_factor_db", 99.0) or 99.0)
    peak_absorb_events: list[dict[str, Any]] = []

    for idx in range(3):
        current = trace[-1]
        tp = float(current.get("true_peak_dbtp", -99.0) or -99.0)
        over_db = tp - float(ceiling_db)
        if over_db <= 1.25:
            break
        drive = float(np.clip((0.65 + over_db * 0.22) * peak_absorb_drive_scale, 0.45, 3.25))
        mix = float(np.clip((0.22 + over_db * 0.055) * peak_absorb_mix_scale, 0.12, 0.78))
        out = soft_clip(out, drive_db=drive, mix=mix)
        an = snap(f"peak_absorb_soft_clip_{idx+1}", out)
        peak_absorb_events.append({
            "pass": idx + 1,
            "input_over_ceiling_db": round(float(over_db), 3),
            "drive_db": round(drive, 3),
            "mix": round(mix, 3),
            "result_lufs": round(float(an.get("integrated_lufs", -99.0) or -99.0), 3),
            "result_true_peak_db": round(float(an.get("approx_true_peak_dbfs", -99.0) or -99.0), 3),
        })

    limited = oversampled_limit_chunked(
        out,
        sr,
        ceiling_db=ceiling_db,
        oversample=os_factor,
        lookahead_ms=limiter_lookahead_ms,
        release_ms=limiter_release_ms,
    )
    snap("after_oversampled_candidate_limiter", limited)
    normalized, tp_gain = _normalize_to_true_peak(limited, ceiling_db, sr=sr, oversample=os_factor)
    norm_an = snap("after_candidate_true_peak_normalize", normalized)
    out = normalized

    recovery_events: list[dict[str, Any]] = []
    best = out
    best_an = norm_an
    best_lufs = float(norm_an.get("integrated_lufs", -99.0) or -99.0)
    prev_lufs = best_lufs
    for idx in range(density_recovery_max_passes if density_recovery_allowed else 0):
        if best_lufs >= min(float(target_lufs) - 0.15, float(floor_lufs) + 0.75):
            break
        gap = max(0.0, min(float(target_lufs), float(floor_lufs) + 1.35) - best_lufs)
        if gap < 0.18:
            break
        add_gain = float(np.clip(gap * 0.72 * density_recovery_gain_scale, 0.10, 1.65))
        trial = apply_gain_db(best, add_gain)
        clip_drive = float(np.clip((0.85 + add_gain * 0.65 + idx * 0.20) * density_recovery_drive_scale, 0.35, 3.60))
        clip_mix = float(np.clip((0.30 + add_gain * 0.10 + idx * 0.055) * density_recovery_mix_scale, 0.08, 0.82))
        trial = soft_clip(trial, drive_db=clip_drive, mix=clip_mix)
        trial = oversampled_limit_chunked(
            trial,
            sr,
            ceiling_db=ceiling_db,
            oversample=os_factor,
            lookahead_ms=density_recovery_limiter_lookahead_ms,
            release_ms=density_recovery_limiter_release_ms,
        )
        trial, trial_tp_gain = _normalize_to_true_peak(trial, ceiling_db, sr=sr, oversample=os_factor)
        trial_an = analyze_audio_fast_qc(trial, sr, true_peak_oversample=os_factor)
        trial_lufs = float(trial_an.get("integrated_lufs", -99.0) or -99.0)
        trial_crest = float(trial_an.get("crest_factor_db", 99.0) or 99.0)
        trial_warnings = _safety_risk(before, trial_an, decision)
        trial_hard = _commercial_loudness_candidate_hard_blocks(before, trial_an, decision, profile, trial_warnings)
        trial_problem = _candidate_problem_set(before, trial_an, decision, profile, trial_warnings, base_problem_set=[])
        improved = trial_lufs > prev_lufs + 0.12
        strict_ok = bool(improved and not trial_hard and not trial_problem.get("critical_problem_set"))
        event = {
            "pass": idx + 1,
            "add_gain_db": round(add_gain, 3),
            "clip_drive_db": round(clip_drive, 3),
            "clip_mix": round(clip_mix, 3),
            "true_peak_normalize_gain_db": round(float(trial_tp_gain), 3),
            "result_lufs": round(trial_lufs, 3),
            "result_crest_db": round(trial_crest, 3),
            "warnings": trial_warnings,
            "hard_blocks": trial_hard,
            "problem_set": trial_problem,
            "accepted": bool(strict_ok),
            "stop_if_rejected_reason": "no_lufs_improvement_or_unresolved_problem_set",
        }
        recovery_events.append(event)
        trace.append({"stage": f"density_recovery_{idx+1}", **_analysis_summary_for_loudness(trial_an)})
        if strict_ok:
            best = trial
            best_an = trial_an
            best_lufs = trial_lufs
            prev_lufs = trial_lufs
        else:
            break

    limiter_loss_db = float(start_lufs - best_lufs) if np.isfinite(start_lufs) and np.isfinite(best_lufs) else 0.0
    report = {
        "schema_version": "busy_hot_render_attempt_trace_v8_5_3_27",
        "ceiling_db": round(float(ceiling_db), 3),
        "target_lufs": round(float(target_lufs), 3),
        "commercial_floor_lufs": round(float(floor_lufs), 3),
        "start_lufs_after_gain": round(float(start_lufs), 3),
        "start_true_peak_after_gain_db": round(float(start_tp), 3),
        "start_crest_after_gain_db": round(float(start_crest), 3),
        "peak_absorb_events": peak_absorb_events,
        "initial_limiter_true_peak_normalize_gain_db": round(float(tp_gain), 3),
        "density_recovery_events": recovery_events,
        "limiter_loudness_loss_db": round(float(limiter_loss_db), 3),
        "limiter_overattenuation_detected": bool(limiter_loss_db > 2.75 and start_lufs > float(floor_lufs) - 0.5),
        "strict_problem_policy": "density recovery stops on any unresolved critical problem set; no accepted_with_warnings for hot recovery",
        "research_planner_render_controls": render_controls,
        "v55_dml_lite_render_controls": {
            "density_recovery_limiter_release_ms": round(float(density_recovery_limiter_release_ms), 3),
            "density_recovery_limiter_lookahead_ms": round(float(density_recovery_limiter_lookahead_ms), 3),
            "v55_density_lift_scale": round(float(v55_density_lift_scale), 4),
            "active": bool(render_controls.get("v55_ivrcf_controlled_direct_activation") or abs(v55_density_lift_scale - 1.0) > 1e-6),
        },
        "trace": trace,
        "final_analysis_full": best_an,
        "final_analysis": _analysis_summary_for_loudness(best_an),
    }
    return np.nan_to_num(best, nan=0.0, posinf=0.0, neginf=0.0), report




def _analysis_lra_lu(analysis: dict[str, Any] | None) -> float | None:
    if not isinstance(analysis, dict):
        return None
    for key in ("lra_lu", "loudness_range_lu"):
        try:
            val = analysis.get(key)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    try:
        st = analysis.get("loudness", {}).get("short_term_lufs", {})
        val = st.get("lra_lu")
        if val is not None and np.isfinite(float(val)):
            return float(val)
        p95 = st.get("p95")
        p10 = st.get("p10")
        if p95 is not None and p10 is not None:
            return max(0.0, float(p95) - float(p10))
    except Exception:
        pass
    return None


def _lra_target_for_decision(decision: dict[str, Any] | None, mode: str = "") -> dict[str, Any]:
    """Operational LRA target ranges by genre/mode.

    These are mastering guardrails, not hard broadcast standards. They prevent
    the loudness prep/limiter from making club tracks lifelessly flat or leaving
    dynamic/live material over-squashed.
    """
    decision = decision or {}
    txt = _profile_text(decision)
    mode_l = str(mode or "").lower()
    if any(k in txt for k in ["spoken", "podcast", "narration"]):
        lo, hi, name = 3.0, 8.0, "spoken_word"
    elif any(k in txt for k in ["cinematic", "orchestral", "score"]):
        lo, hi, name = 8.0, 16.0, "cinematic_orchestral"
    elif any(k in txt for k in ["live", "acoustic", "ballad", "worship"]):
        lo, hi, name = 6.0, 12.0, "live_acoustic_vocal"
    elif any(k in txt for k in ["lofi", "lo_fi", "vintage"]):
        lo, hi, name = 5.0, 10.0, "lofi_vintage"
    elif any(k in txt for k in ["hard_techno", "schranz", "industrial", "hard dance"]):
        lo, hi, name = 2.5, 5.5, "hard_dance_techno"
    elif any(k in txt for k in ["drift", "phonk", "trap", "hiphop", "808", "bass"]):
        lo, hi, name = 3.5, 7.0, "bass_phonk_trap"
    elif any(k in txt for k in ["edm", "festival", "house", "bounce", "club", "garage"]):
        lo, hi, name = 3.5, 6.5, "edm_club"
    elif any(k in txt for k in ["kpop", "k-pop", "pop", "rnb", "commercial"]):
        lo, hi, name = 4.0, 8.0, "commercial_pop"
    else:
        lo, hi, name = 4.0, 9.0, "universal_safe"

    if "hot" in mode_l:
        lo = max(2.0, lo - 0.7)
        hi = max(lo + 1.5, hi - 0.7)
    elif "streaming" in mode_l or "safe" in mode_l or "clean" in mode_l:
        lo += 0.5
        hi += 0.8

    return {"profile": name, "min_lra_lu": round(lo, 2), "max_lra_lu": round(hi, 2), "target_mid_lra_lu": round((lo + hi) / 2.0, 2)}




def _lra_guard_thresholds() -> dict[str, float]:
    strictness = str(os.environ.get("BUSY_LRA_STRICTNESS", "normal")).strip().lower()
    if strictness in {"strict", "tight"}:
        return {"flat_margin": 0.10, "wide_margin": 0.40, "delta_floor": -2.8}
    if strictness in {"loose", "relaxed"}:
        return {"flat_margin": 0.60, "wide_margin": 1.20, "delta_floor": -4.8}
    return {"flat_margin": 0.25, "wide_margin": 0.75, "delta_floor": -3.5}

def _lra_guard_summary(before: dict[str, Any] | None, after: dict[str, Any] | None, decision: dict[str, Any] | None, mode: str = "") -> dict[str, Any]:
    target = _lra_target_for_decision(decision or {}, mode)
    in_lra = _analysis_lra_lu(before)
    out_lra = _analysis_lra_lu(after)
    warnings: list[str] = []
    th = _lra_guard_thresholds()
    if out_lra is not None:
        if out_lra < float(target["min_lra_lu"]) - th["flat_margin"]:
            warnings.append("lra_too_flat")
        if out_lra > float(target["max_lra_lu"]) + th["wide_margin"]:
            warnings.append("lra_too_wide")
    if in_lra is not None and out_lra is not None:
        delta = out_lra - in_lra
        if delta < th["delta_floor"]:
            warnings.append("lra_reduced_too_much")
    else:
        delta = None
    return {
        "active": _env_bool("BUSY_LRA_GUARD", True),
        "target": target,
        "strictness": str(os.environ.get("BUSY_LRA_STRICTNESS", "normal")).strip().lower(),
        "thresholds": _lra_guard_thresholds(),
        "input_lra_lu": round(in_lra, 3) if in_lra is not None else None,
        "output_lra_lu": round(out_lra, 3) if out_lra is not None else None,
        "lra_delta_lu": round(delta, 3) if delta is not None else None,
        "warnings": warnings if _env_bool("BUSY_LRA_GUARD", True) else [],
    }


def _airqc_num(x: Any, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def _analysis_true_peak_authority(analysis: dict[str, Any] | None) -> float | None:
    a = analysis if isinstance(analysis, dict) else {}
    loud = a.get("loudness", {}) if isinstance(a.get("loudness"), dict) else {}
    for key in ("true_peak_dbtp", "true_peak_dbfs", "approx_true_peak_dbfs"):
        v = _airqc_num(a.get(key), None)
        if v is not None:
            return v
    for key in ("true_peak_dbtp", "true_peak_dbfs", "approx_true_peak_dbfs"):
        v = _airqc_num(loud.get(key), None)
        if v is not None:
            return v
    return None


def _analysis_band_db(analysis: dict[str, Any] | None, *keys: str) -> float | None:
    a = analysis if isinstance(analysis, dict) else {}
    bands = a.get("bands_db") if isinstance(a.get("bands_db"), dict) else a.get("bands") if isinstance(a.get("bands"), dict) else {}
    q = a.get("quality_indices", {}) if isinstance(a.get("quality_indices"), dict) else {}
    for k in keys:
        v = _airqc_num(a.get(k), None)
        if v is not None:
            return v
        v = _airqc_num(q.get(k), None)
        if v is not None:
            return v
        v = _airqc_num(bands.get(k), None)
        if v is not None:
            return v
    return None


def _analysis_ratio(analysis: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    a = analysis if isinstance(analysis, dict) else {}
    q = a.get("quality_indices", {}) if isinstance(a.get("quality_indices"), dict) else {}
    ms = a.get("ms", {}) if isinstance(a.get("ms"), dict) else {}
    for k in keys:
        for src in (a, q, ms):
            v = _airqc_num(src.get(k), None) if isinstance(src, dict) else None
            if v is not None:
                return v
    return default


def _airqc4_class_score(label: str | None) -> int:
    return {"Safe": 0, "Acceptable": 1, "Caution": 2, "Warning": 3, "High-Risk": 4, "Reject": 5}.get(str(label or ""), 2)


def _airqc4_label_from_thresholds(value: float | None, thresholds: list[float]) -> str:
    if value is None or not np.isfinite(float(value)):
        return "Unknown"
    v = max(0.0, float(value))
    t = [max(0.0, float(x)) for x in thresholds]
    if len(t) < 5:
        return "Unknown"
    if v <= t[0]:
        return "Safe"
    if v <= t[1]:
        return "Acceptable"
    if v <= t[2]:
        return "Caution"
    if v <= t[3]:
        return "Warning"
    if v <= t[4]:
        return "High-Risk"
    return "Reject"


def _airqc4_max_label(*labels: str) -> str:
    vals = [( _airqc4_class_score(x), x) for x in labels if x and x != "Unknown"]
    if not vals:
        return "Unknown"
    return max(vals, key=lambda x: x[0])[1]


def _airqc4_shift_thresholds(thresholds: list[float], delta: float = 0.0, floor: float = 0.05) -> list[float]:
    shifted = [max(floor, float(x) + float(delta)) for x in thresholds]
    # Keep the sequence monotonic even after a negative source modifier.
    out: list[float] = []
    last = 0.0
    for v in shifted:
        v = max(v, last + 0.02)
        out.append(v)
        last = v
    return out


def _airqc4_analysis_lufs(analysis: dict[str, Any] | None) -> float | None:
    a = analysis if isinstance(analysis, dict) else {}
    loud = a.get("loudness", {}) if isinstance(a.get("loudness"), dict) else {}
    for src in (a, loud):
        for key in ("integrated_lufs", "stereo_integrated_lufs", "lufs"):
            v = _airqc_num(src.get(key), None) if isinstance(src, dict) else None
            if v is not None:
                return v
    return None


def _airqc4_band_value(analysis: dict[str, Any] | None, *keys: str, channel: str = "broadband") -> float | None:
    a = analysis if isinstance(analysis, dict) else {}
    band_energy = a.get("band_energy_db") if isinstance(a.get("band_energy_db"), dict) else {}
    bands_db = a.get("bands_db") if isinstance(a.get("bands_db"), dict) else {}
    q = a.get("quality_indices") if isinstance(a.get("quality_indices"), dict) else {}
    ms = a.get("ms") if isinstance(a.get("ms"), dict) else {}
    if channel == "mid":
        ms_bands = ms.get("mid_band_db") if isinstance(ms.get("mid_band_db"), dict) else {}
    elif channel == "side":
        ms_bands = ms.get("side_band_db") if isinstance(ms.get("side_band_db"), dict) else {}
    else:
        ms_bands = {}
    for key in keys:
        for src in (band_energy, bands_db, ms_bands, q, a):
            v = _airqc_num(src.get(key), None) if isinstance(src, dict) else None
            if v is not None:
                return v
    return None


def _airqc4_relative_band_delta(before: dict[str, Any] | None, after: dict[str, Any] | None, loudness_gain: float | None, *keys: str, channel: str = "broadband") -> float | None:
    ib = _airqc4_band_value(before, *keys, channel=channel)
    ob = _airqc4_band_value(after, *keys, channel=channel)
    if ib is None or ob is None:
        return None
    lg = float(loudness_gain) if loudness_gain is not None and np.isfinite(float(loudness_gain)) else 0.0
    return float(ob) - float(ib) - lg


def _airqc4_context_modifiers(decision: dict[str, Any] | None, mode: str = "") -> dict[str, Any]:
    d = decision if isinstance(decision, dict) else {}
    try:
        import json as _json
        text = (_json.dumps(d, ensure_ascii=False, default=str) + " " + str(mode)).lower()
    except Exception:
        text = str(mode).lower()
    mods = {
        "lra_loss_lu": 0.0,
        "crest_loss_db": 0.0,
        "low_end_db": 0.0,
        "width_class_shift": 0,
        "hf_harshness_db": 0.0,
        "notes": [],
    }
    def _apply(name: str, lra: float = 0.0, crest: float = 0.0, low: float = 0.0, width: int = 0, hf: float = 0.0) -> None:
        mods["lra_loss_lu"] += lra
        mods["crest_loss_db"] += crest
        mods["low_end_db"] += low
        mods["width_class_shift"] += width
        mods["hf_harshness_db"] += hf
        mods["notes"].append(name)
    if any(x in text for x in ("edm", "trap", "hyperpop", "club", "dance", "phonk")):
        _apply("loud_electronic_tolerance", lra=0.5, crest=0.5, width=1, hf=-0.25)
    elif any(x in text for x in ("rock", "metal", "punk", "indie")):
        _apply("rock_metal_transient_tolerance", lra=0.25, crest=0.5, width=-1, hf=-0.5)
    elif any(x in text for x in ("hip", "rap", "808")):
        _apply("hiphop_low_end_translation_priority", lra=0.25, crest=0.5, low=-0.5, width=-1)
    elif any(x in text for x in ("ballad", "acoustic", "folk", "jazz", "singer")):
        _apply("natural_vocal_dynamic_preservation", lra=-0.5, crest=-0.5, width=-1, hf=-0.25)
    elif any(x in text for x in ("cinematic", "orchestral", "ambient", "classical")):
        _apply("cinematic_loudness_chase_suppression", lra=-1.0, crest=-1.0, width=-1, hf=-0.5)
    if any(x in text for x in ("suno", "ai_generated", "ai-generated", "ai generated", "artifact")):
        _apply("ai_generated_artifact_conservative", lra=-0.25, crest=-0.25, low=-0.25, width=-1, hf=-1.0)
    mods["width_class_shift"] = int(np.clip(int(mods.get("width_class_shift") or 0), -2, 2))
    return mods


def _airqc4_shift_class(label: str, shift: int = 0) -> str:
    labels = ["Safe", "Acceptable", "Caution", "Warning", "High-Risk", "Reject"]
    if label not in labels:
        return label
    idx = int(np.clip(labels.index(label) + int(shift), 0, len(labels) - 1))
    return labels[idx]


def _bamix_v6319_policy_for_adaptive_qc(decision: dict[str, Any] | None) -> dict[str, Any]:
    """Return BAMix handoff policy for adaptive QC without relying on env.

    v63.1.9 made BAMix handoff a readiness-aware governance layer.  The
    adaptive input-relative QC must therefore compare the delivered master
    against the effective BAMix premaster reference, not an older raw/pre-clean
    stereo snapshot that may have a different crest/LRA baseline.
    """
    d = decision if isinstance(decision, dict) else {}
    direct = d.get("busy_auto_mixing_mastering_handoff_policy")
    if isinstance(direct, dict) and direct.get("active"):
        return direct
    ai = d.get("auto_intensity_policy") if isinstance(d.get("auto_intensity_policy"), dict) else {}
    hp = ai.get("bamix_premaster_handoff_policy") if isinstance(ai, dict) else None
    if isinstance(hp, dict) and hp.get("active"):
        return hp
    return {}


def _v62_9_9_4_input_relative_damage_scoring(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    mode: str,
    *,
    lra_loss: float | None = None,
    crest_loss: float | None = None,
    in_lra: float | None = None,
    in_crest: float | None = None,
    in_tp: float | None = None,
    out_tp: float | None = None,
) -> dict[str, Any]:
    """v62.9.9.4: implementation-oriented input-relative damage score.

    This is a report/candidate-arbitration layer only.  It turns the existing
    v62.9.9.3.x input-vs-delivered-final measurements into axis classes and a
    weighted risk score based on the input-relative criteria report.  It does
    not add a new render loop, AI judge, stem audio path, or post-process
    expansion.
    """
    enabled = bool(_env_bool("BUSY_V6299_4_INPUT_RELATIVE_DAMAGE_SCORING", True))
    b = before if isinstance(before, dict) else {}
    a = after if isinstance(after, dict) else {}
    mods = _airqc4_context_modifiers(decision, mode)
    bamix_qc_policy = _bamix_v6319_policy_for_adaptive_qc(decision)
    bamix_qc_mode = str(bamix_qc_policy.get("handoff_mode") or bamix_qc_policy.get("profile") or "").lower()
    bamix_density_qc = bool(bamix_qc_policy.get("active") and "density" in bamix_qc_mode)
    bamix_max_crest_loss = _airqc_num(bamix_qc_policy.get("max_crest_loss_db"), None) if bamix_density_qc else None
    bamix_floor_lufs = _airqc_num(bamix_qc_policy.get("effective_floor_lufs"), None) if bamix_density_qc else None
    bamix_target_lufs = _airqc_num(bamix_qc_policy.get("effective_final_target_lufs"), None) if bamix_density_qc else None

    in_lufs = _airqc4_analysis_lufs(b)
    out_lufs = _airqc4_analysis_lufs(a)
    out_lra = _analysis_lra_lu(a)
    loud_gain = float(out_lufs) - float(in_lufs) if in_lufs is not None and out_lufs is not None else None
    bamix_density_floor_met = bool(bamix_density_qc and out_lufs is not None and bamix_floor_lufs is not None and float(out_lufs) >= float(bamix_floor_lufs) - 0.12)

    # 1) LRA loss: input-bin specific thresholds from the research table.
    lra_val = max(0.0, float(lra_loss)) if lra_loss is not None else None
    if in_lra is None:
        lra_thresholds = [0.4, 0.8, 1.2, 1.8, 2.5]
    elif float(in_lra) >= 8.0:
        lra_thresholds = [1.0, 2.0, 3.0, 4.0, 4.5]
    elif float(in_lra) >= 4.0:
        lra_thresholds = [0.8, 1.5, 2.5, 3.5, 4.0]
    elif float(in_lra) >= 2.0:
        lra_thresholds = [0.4, 0.8, 1.2, 1.8, 2.5]
    elif float(in_lra) >= 1.5:
        lra_thresholds = [0.3, 0.5, 0.8, 1.2, 1.5]
    else:
        lra_thresholds = [0.05, 0.2, 0.4, 0.6, 0.8]
    lra_class = _airqc4_label_from_thresholds(lra_val, _airqc4_shift_thresholds(lra_thresholds, float(mods.get("lra_loss_lu") or 0.0)))
    if bamix_density_qc and lra_val is not None and out_lra is not None:
        # v63.1.9.3: density-completion intentionally spends more LRA than
        # transparent polish.  If the delivered master has reached the BAMix
        # commercial floor and retains a usable final LRA, keep this axis as
        # monitored cost instead of letting the legacy table imply a failed
        # transparent-polish handoff.
        bamix_lra_cost_limit = max(2.2, min(3.1, float(in_lra or 0.0) * 0.55)) if in_lra is not None else 2.6
        if bamix_density_floor_met and float(out_lra) >= 2.45 and float(lra_val) <= bamix_lra_cost_limit:
            lra_class = "Caution" if _airqc4_class_score(lra_class) > _airqc4_class_score("Caution") else lra_class

    # 2) Crest loss: input-crest specific thresholds.
    crest_val = max(0.0, float(crest_loss)) if crest_loss is not None else None
    if in_crest is None:
        crest_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    elif float(in_crest) >= 12.0:
        crest_thresholds = [0.7, 1.2, 1.8, 2.5, 3.5]
    elif float(in_crest) >= 9.0:
        crest_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5]
    elif float(in_crest) >= 7.0:
        crest_thresholds = [0.4, 0.7, 1.2, 1.6, 2.0]
    else:
        crest_thresholds = [0.3, 0.5, 0.8, 1.2, 1.5]
    shifted_crest_thresholds = _airqc4_shift_thresholds(crest_thresholds, float(mods.get("crest_loss_db") or 0.0))
    if bamix_density_qc and bamix_max_crest_loss is not None and crest_val is not None:
        bamix_crest_grace = float(_env_float("BUSY_BAMIX_V6319_ADAPTIVE_QC_CREST_GRACE_DB", 0.18, 0.0, 0.6))
        # In density-completion mode the BAMix handoff policy is the crest budget
        # authority.  A tiny boundary overshoot should not become a generic
        # Reject axis while the delivered master reached the commercial floor
        # and the existing v57/v60/v61/export guards accepted it.
        if float(crest_val) <= float(bamix_max_crest_loss) + bamix_crest_grace:
            crest_class = "Warning" if float(crest_val) > float(bamix_max_crest_loss) else "Caution"
            shifted_crest_thresholds = [
                max(0.05, min(1.2, float(bamix_max_crest_loss) * 0.25)),
                max(0.10, min(2.2, float(bamix_max_crest_loss) * 0.45)),
                max(0.15, min(3.4, float(bamix_max_crest_loss) * 0.70)),
                float(bamix_max_crest_loss) + bamix_crest_grace,
                float(bamix_max_crest_loss) + max(0.45, bamix_crest_grace + 0.25),
            ]
        else:
            crest_class = _airqc4_label_from_thresholds(crest_val, shifted_crest_thresholds)
    else:
        crest_class = _airqc4_label_from_thresholds(crest_val, shifted_crest_thresholds)

    # 3) Loudness-gain vs dynamic-cost tradeoff.
    dynamic_cost = None
    cost_per_lu = None
    lufs_class = "Unknown"
    if loud_gain is not None:
        ll = max(0.0, float(lra_val or 0.0))
        cl = max(0.0, float(crest_val or 0.0))
        dynamic_cost = 0.6 * ll + 0.4 * cl
        cost_per_lu = dynamic_cost / max(float(loud_gain), 0.5)
        if float(loud_gain) < 0.5:
            lufs_class = "Reject" if dynamic_cost > 1.5 else "Safe"
        elif float(loud_gain) < 1.5:
            lufs_class = "Acceptable" if ll <= 0.5 and cl <= 0.7 else "Warning"
        elif float(loud_gain) < 3.0:
            lufs_class = "Acceptable" if ll <= 1.0 and cl <= 1.2 else "Caution" if ll <= 1.4 and cl <= 1.8 else "Warning"
        elif float(loud_gain) < 4.0:
            lufs_class = "Caution" if ll <= 1.2 and cl <= 1.5 and (out_tp is None or float(out_tp) <= -1.0) else "Warning"
        else:
            lufs_class = "Warning" if dynamic_cost <= 2.5 else "High-Risk"
        if cost_per_lu is not None and cost_per_lu > 0.9:
            lufs_class = _airqc4_max_label(lufs_class, "Warning")
        if bamix_density_floor_met and cost_per_lu is not None and float(cost_per_lu) <= 0.68 and dynamic_cost is not None and float(dynamic_cost) <= 4.1:
            # Large LUFS gain is expected in density-completion mode; judge it
            # by cost-per-LU and existing safety gates rather than the old raw
            # input tradeoff table alone.
            lufs_class = "Caution" if _airqc4_class_score(lufs_class) > _airqc4_class_score("Caution") else lufs_class

    # 4) Delivered true-peak and input-relative pressure.
    tp_delta = float(out_tp) - float(in_tp) if in_tp is not None and out_tp is not None else None
    final_tp_class = "Unknown"
    if out_tp is not None:
        tp = float(out_tp)
        if tp <= -1.2:
            final_tp_class = "Safe"
        elif tp <= -1.0:
            final_tp_class = "Acceptable"
        elif tp <= -0.5:
            final_tp_class = "Caution"
        elif tp <= 0.0:
            final_tp_class = "Warning"
        else:
            final_tp_class = "Reject"
        # Hot delivered masters should keep extra codec headroom.
        if out_lufs is not None and float(out_lufs) > -11.0 and tp > -1.5:
            final_tp_class = _airqc4_max_label(final_tp_class, "Caution")
    headroom_class = "Unknown"
    if tp_delta is not None:
        if tp_delta <= 1.0:
            headroom_class = "Safe"
        elif tp_delta <= 2.0:
            headroom_class = "Acceptable"
        elif tp_delta <= 3.0:
            headroom_class = "Caution"
        else:
            headroom_class = "Warning"
    tp_class = _airqc4_max_label(final_tp_class, headroom_class)

    # 5) Spectral/translation axes use loudness-normalized relative band deltas.
    sub_delta = _airqc4_relative_band_delta(b, a, loud_gain, "sub_20_60", "b20_80")
    bass_delta = _airqc4_relative_band_delta(b, a, loud_gain, "bass_60_120", "b80_200")
    lowmid_delta = _airqc4_relative_band_delta(b, a, loud_gain, "lowmid_120_500", "b200_500")
    low_vals = [abs(x) for x in (sub_delta, bass_delta, lowmid_delta) if x is not None]
    low_max = max(low_vals) if low_vals else None
    low_class = _airqc4_label_from_thresholds(low_max, _airqc4_shift_thresholds([0.8, 1.2, 1.8, 2.5, 3.0], float(mods.get("low_end_db") or 0.0)))
    if lowmid_delta is not None and -1.5 <= float(lowmid_delta) <= -0.5 and (sub_delta is None or float(sub_delta) >= -1.2) and (bass_delta is None or float(bass_delta) >= -1.2):
        low_class = _airqc4_max_label("Safe", low_class if _airqc4_class_score(low_class) > 2 else "Safe")

    presence_delta = _airqc4_relative_band_delta(b, a, loud_gain, "presence_2000_5000", "b2500_5000", channel="mid")
    frontness_loss = max(0.0, -float(presence_delta)) if presence_delta is not None else None
    front_class = _airqc4_label_from_thresholds(frontness_loss, [0.5, 0.8, 1.5, 2.5, 3.0])

    in_corr = _analysis_ratio(b, "correlation", "corr", "correlation_avg", default=None)
    out_corr = _analysis_ratio(a, "correlation", "corr", "correlation_avg", default=None)
    if in_corr is None:
        in_corr = _airqc4_band_value(b, "correlation_avg", channel="mid")
    if out_corr is None:
        out_corr = _airqc4_band_value(a, "correlation_avg", channel="mid")
    corr_drop = float(in_corr) - float(out_corr) if in_corr is not None and out_corr is not None else None
    in_smr = _analysis_ratio(b, "side_ratio_8k_14k", "side_ratio_highband", "side_ratio_high", default=None)
    out_smr = _analysis_ratio(a, "side_ratio_8k_14k", "side_ratio_highband", "side_ratio_high", default=None)
    width_delta = abs(float(out_smr) - float(in_smr)) if in_smr is not None and out_smr is not None else None
    mono_in = _airqc_num((b.get("loudness") or {}).get("mono_downmix_lufs") if isinstance(b.get("loudness"), dict) else b.get("mono_downmix_lufs"), None)
    mono_out = _airqc_num((a.get("loudness") or {}).get("mono_downmix_lufs") if isinstance(a.get("loudness"), dict) else a.get("mono_downmix_lufs"), None)
    mono_delta = abs((float(mono_out) - float(mono_in)) - float(loud_gain or 0.0)) if mono_in is not None and mono_out is not None else None
    stereo_class = _airqc4_max_label(
        _airqc4_label_from_thresholds((width_delta * 10.0) if width_delta is not None else None, [1.0, 2.0, 3.5, 5.0, 6.0]),
        _airqc4_label_from_thresholds(mono_delta, [0.75, 1.25, 2.0, 3.0, 3.5]),
        _airqc4_label_from_thresholds(corr_drop, [0.05, 0.10, 0.18, 0.25, 0.30]),
    )
    stereo_class = _airqc4_shift_class(stereo_class, -int(mods.get("width_class_shift") or 0) if int(mods.get("width_class_shift") or 0) < 0 else 0)

    harsh_delta = _airqc4_relative_band_delta(b, a, loud_gain, "sibilance_5000_8000", "brilliance_7000_12000", "harshness_index_db")
    air_delta = _airqc4_relative_band_delta(b, a, loud_gain, "air_8000_14000", "air_12000_18000")
    air_loss = max(0.0, -float(air_delta)) if air_delta is not None else None
    hf_class = _airqc4_max_label(
        _airqc4_label_from_thresholds(max(0.0, float(harsh_delta)) if harsh_delta is not None else None, _airqc4_shift_thresholds([0.5, 1.0, 1.75, 2.5, 3.0], float(mods.get("hf_harshness_db") or 0.0))),
        _airqc4_label_from_thresholds(air_loss, [1.0, 1.5, 2.5, 3.5, 4.0]),
    )
    if out_tp is not None and float(out_tp) > -1.0 and harsh_delta is not None and harsh_delta > 2.0:
        hf_class = _airqc4_max_label(hf_class, "Warning")

    # The PDF/schema uses the same class enum for every axis.  When objective
    # PEAQ/ViSQOL-style perceptual metrics are unavailable, keep the axis in
    # the report for schema stability but make it zero-confidence so it is
    # ignored by the weighted score instead of emitting a non-enum "Unknown"
    # class.
    perceptual_class = "Safe"
    confidence_perceptual = 0.0

    axis_specs = [
        ("lra_loss", lra_val, lra_class, 0.95, {"input_lra_lu": in_lra, "thresholds_lu": lra_thresholds, "bamix_handoff_mode": bamix_qc_mode or None, "bamix_density_floor_met": bool(bamix_density_floor_met) if bamix_density_qc else None}),
        ("crest_loss", crest_val, crest_class, 0.95, {"input_crest_db": in_crest, "thresholds_db": shifted_crest_thresholds, "bamix_handoff_mode": bamix_qc_mode or None, "bamix_policy_max_crest_loss_db": round(float(bamix_max_crest_loss), 3) if bamix_max_crest_loss is not None else None}),
        ("lufs_tradeoff", {"loudness_gain_lu": loud_gain, "dynamic_cost": dynamic_cost, "cost_per_lu": cost_per_lu}, lufs_class, 0.9 if loud_gain is not None else 0.0, {"bamix_handoff_mode": bamix_qc_mode or None, "bamix_density_floor_met": bool(bamix_density_floor_met) if bamix_density_qc else None}),
        ("true_peak_pressure", {"final_true_peak_dbtp": out_tp, "headroom_consumed_db": tp_delta}, tp_class, 0.95 if out_tp is not None else 0.0, {}),
        ("low_end_low_mid_delta", {"sub_delta_db": sub_delta, "bass_delta_db": bass_delta, "lowmid_delta_db": lowmid_delta}, low_class, 0.75 if low_max is not None else 0.0, {}),
        ("vocal_frontness_loss", {"presence_relative_delta_db": presence_delta, "frontness_loss_db": frontness_loss}, front_class, 0.55 if frontness_loss is not None else 0.0, {"policy": "proxy only unless lead confidence is available"}),
        ("stereo_mono_delta", {"width_ratio_delta": width_delta, "mono_delta_db": mono_delta, "correlation_drop": corr_drop}, stereo_class, 0.7 if stereo_class != "Unknown" else 0.0, {}),
        ("hf_codec_risk", {"harshness_relative_delta_db": harsh_delta, "air_loss_db": air_loss}, hf_class, 0.75 if hf_class != "Unknown" else 0.0, {}),
        ("perceptual_final", None, perceptual_class, confidence_perceptual, {"policy": "objective proxy unavailable in this lightweight pass; external/AI review remains advisory"}),
    ]
    axis_results: list[dict[str, Any]] = []
    weights = {
        "lra_loss": 1.0,
        "crest_loss": 1.2,
        "lufs_tradeoff": 1.2,
        "true_peak_pressure": 1.3,
        "low_end_low_mid_delta": 1.1,
        "vocal_frontness_loss": 1.1,
        "stereo_mono_delta": 1.0,
        "hf_codec_risk": 1.2,
        "perceptual_final": 0.8,
    }
    total = 0.0
    total_w = 0.0
    high_risk_count = 0
    reject_count = 0
    valid_axis_classes = {"Safe", "Acceptable", "Caution", "Warning", "High-Risk", "Reject"}
    for axis, value, label, conf, notes in axis_specs:
        raw_label = str(label or "")
        note_obj = dict(notes or {}) if isinstance(notes, dict) else {"notes": notes}
        c = float(conf or 0.0)
        if raw_label not in valid_axis_classes:
            note_obj["measurement_status"] = "unavailable"
            note_obj["raw_class"] = raw_label or None
            note_obj["scoring_policy"] = "zero_confidence_axis_ignored"
            label = "Safe"
            c = 0.0
        else:
            label = raw_label
            note_obj.setdefault("measurement_status", "available" if c > 0.0 else "zero_confidence")
        score = _airqc4_class_score(label)
        if label == "High-Risk":
            high_risk_count += 1
        if label == "Reject":
            reject_count += 1
        if c > 0.0:
            total += float(weights.get(axis, 1.0)) * float(score) * c
            total_w += float(weights.get(axis, 1.0)) * c
        axis_results.append({
            "axis": axis,
            "value": value,
            "class": label,
            "score": score,
            "confidence": round(c, 3),
            "notes": note_obj,
        })
    weighted_risk = total / max(total_w, 1e-9) if total_w > 0 else None

    hard_veto = False
    hard_reasons: list[str] = []
    if out_tp is not None and float(out_tp) > -1.0 and str(mode).lower() in {"streaming_conservative", "streaming_loud", "broadcast", ""}:
        # Treat this as a candidate-selection hard veto for delivery profiles.
        # The export-lock/final limiter remains the measurement authority for the
        # actual delivered final, but a candidate above the delivery ceiling should
        # not be preferred by the adaptive scorer.
        hard_veto = True
        hard_reasons.append("delivered_true_peak_above_minus_1_dbTP_reference")
    if reject_count >= 1:
        hard_veto = True
        hard_reasons.append("one_or_more_axis_reject")
    if high_risk_count + reject_count >= 2:
        hard_veto = True
        hard_reasons.append("multiple_high_risk_axes")
    if loud_gain is not None and loud_gain < 0.5 and dynamic_cost is not None and dynamic_cost > 1.5:
        hard_veto = True
        hard_reasons.append("minimal_loudness_benefit_excessive_dynamic_cost")

    if not enabled:
        acceptance = "disabled"
        recommended_action = "no_action"
        selected_mode = "PRESERVE"
    elif hard_veto:
        acceptance = "reject_or_bypass"
        recommended_action = "reject_louder_candidate_or_keep_cleaner_lower_loudness_candidate"
        selected_mode = "BYPASS"
    elif weighted_risk is None:
        acceptance = "insufficient_metrics"
        recommended_action = "retain_existing_safety_governance"
        selected_mode = "CONTROL"
    elif weighted_risk < 1.0:
        acceptance = "accept"
        recommended_action = "accept_delivered_final"
        selected_mode = "ENHANCE" if (loud_gain is not None and loud_gain >= 1.0) else "PRESERVE"
    elif weighted_risk < 1.6:
        acceptance = "accept_with_caution"
        recommended_action = "accept_but_monitor_axis_warnings"
        selected_mode = "CONTROL"
    elif weighted_risk < 2.2:
        acceptance = "compare_alternates"
        recommended_action = "compare_softer_or_lower_target_candidate"
        selected_mode = "CONTROL"
    elif weighted_risk < 2.8:
        acceptance = "prefer_lower_loudness_candidate"
        recommended_action = "drop_loudness_push_and_re_render_or_keep_cleaner_standard_final"
        selected_mode = "PRESERVE"
    else:
        acceptance = "reject_or_bypass"
        recommended_action = "risk_outweighs_loudness_benefit"
        selected_mode = "BYPASS"

    recommended_warning = None
    if acceptance == "accept_with_caution":
        recommended_warning = "adaptive_qc_accept_with_caution"
    elif acceptance == "compare_alternates":
        recommended_warning = "adaptive_qc_compare_softer_candidate"
    elif acceptance == "prefer_lower_loudness_candidate":
        recommended_warning = "adaptive_qc_prefer_lower_loudness_candidate"
    elif acceptance == "reject_or_bypass":
        recommended_warning = "adaptive_qc_reject_or_bypass"

    return {
        "schema_version": "busy_input_relative_damage_scoring_v8_5_3_62_9_9_4_4",
        "enabled": enabled,
        "policy": "Apply the input-relative criteria table as QC scoring/candidate arbitration, not as a new DSP command or song-specific exception.",
        "source_basis": "Busy Auto Mastering quality evaluation criteria request.pdf / input-relative mastering damage and preservation criteria",
        "genre_source_modifiers": mods,
        "input_metrics": {
            "integrated_lufs": round(float(in_lufs), 3) if in_lufs is not None else None,
            "lra_lu": round(float(in_lra), 3) if in_lra is not None else None,
            "crest_factor_db": round(float(in_crest), 3) if in_crest is not None else None,
            "true_peak_dbtp": round(float(in_tp), 3) if in_tp is not None else None,
        },
        "candidate_metrics": {
            "integrated_lufs": round(float(out_lufs), 3) if out_lufs is not None else None,
            "lra_lu": round(float(_analysis_lra_lu(a)), 3) if _analysis_lra_lu(a) is not None else None,
            "crest_factor_db": round(float(_airqc_num(a.get("crest_factor_db"), 0.0)), 3) if _airqc_num(a.get("crest_factor_db"), None) is not None else None,
            "true_peak_dbtp": round(float(out_tp), 3) if out_tp is not None else None,
        },
        "axis_results": axis_results,
        "weighted_risk": round(float(weighted_risk), 4) if weighted_risk is not None else None,
        "decision": {
            "mode": selected_mode,
            "acceptance": acceptance,
            "weighted_risk": round(float(weighted_risk), 4) if weighted_risk is not None else None,
            "hard_veto": bool(hard_veto),
            "hard_veto_reasons": sorted(set(hard_reasons)),
            "recommended_action": recommended_action,
            "recommended_warning": recommended_warning,
            "recommended_true_peak_ceiling_dbtp": -1.5 if out_lufs is not None and float(out_lufs) > -11.0 else -1.2,
        },
    }


def _adaptive_input_relative_qc_governance(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    mode: str,
    lra_guard: dict[str, Any] | None = None,
    current_reasons: list[str] | None = None,
) -> dict[str, Any]:
    """v62.9.9.3: input-relative QC governance.

    Deep-research reports repeatedly distinguished final absolute thresholds from
    mastering-induced damage.  This report-only governance layer compares input
    and final measurements for LRA, crest, true-peak pressure, low-end/frontness,
    stereo/mono and HF/codec risk.  It does not create a new DSP stage; it only
    reclassifies warnings and supplies candidate-selection/safe-push context.
    """
    enabled = bool(_env_bool("BUSY_V6299_3_ADAPTIVE_INPUT_RELATIVE_QC", True))
    b = before if isinstance(before, dict) else {}
    a = after if isinstance(after, dict) else {}
    lg = lra_guard if isinstance(lra_guard, dict) else {}
    reasons = [str(x) for x in (current_reasons or [])]
    target = lg.get("target") if isinstance(lg.get("target"), dict) else _lra_target_for_decision(decision or {}, mode)
    bamix_qc_policy = _bamix_v6319_policy_for_adaptive_qc(decision)
    bamix_qc_mode = str(bamix_qc_policy.get("handoff_mode") or bamix_qc_policy.get("profile") or "").lower()
    bamix_density_qc = bool(bamix_qc_policy.get("active") and "density" in bamix_qc_mode)
    bamix_max_crest_loss = _airqc_num(bamix_qc_policy.get("max_crest_loss_db"), None) if bamix_density_qc else None
    bamix_floor_lufs = _airqc_num(bamix_qc_policy.get("effective_floor_lufs"), None) if bamix_density_qc else None
    bamix_reference_override: dict[str, Any] = {"active": False}
    _pm_lufs = _airqc_num(bamix_qc_policy.get("premaster_lufs"), None) if bamix_qc_policy.get("active") else None
    if _pm_lufs is not None:
        bamix_reference_override["active"] = True
        bamix_reference_override["integrated_lufs"] = round(float(_pm_lufs), 3)

    in_lra = _analysis_lra_lu(b)
    if bamix_qc_policy.get("active"):
        _pm_lra = _airqc_num(bamix_qc_policy.get("premaster_lra_lu"), None)
        if _pm_lra is not None:
            in_lra = _pm_lra
            bamix_reference_override["active"] = True
            bamix_reference_override["lra_lu"] = round(float(_pm_lra), 3)
    out_lra = _analysis_lra_lu(a)
    if out_lra is None:
        out_lra = _airqc_num(a.get("lra_lu"), None)
    lra_loss = (float(in_lra) - float(out_lra)) if in_lra is not None and out_lra is not None else None
    lra_retention = (float(out_lra) / max(0.25, float(in_lra))) if in_lra is not None and out_lra is not None else None
    target_min = _airqc_num(target.get("min_lra_lu") if isinstance(target, dict) else None, 2.5) or 2.5
    adaptive_floor = None
    if in_lra is not None:
        # Dense/AI-flat sources should be judged by additional loss first;
        # dynamic sources keep a higher adaptive floor.
        adaptive_floor = float(np.clip(min(target_min, max(1.35, float(in_lra) - 0.65)), 1.25, target_min))
    elif out_lra is not None:
        adaptive_floor = float(np.clip(min(target_min, max(1.35, float(out_lra) - 0.20)), 1.25, target_min))

    lra_class = "unknown"
    remove: list[str] = []
    add: list[str] = []
    candidate_rejection_required = False
    safe_push_adjustment = "none"
    if in_lra is not None and out_lra is not None:
        source_flat = float(in_lra) < max(2.5, target_min - 0.75)
        low_final = float(out_lra) < target_min
        severe_loss = bool(lra_loss is not None and lra_loss > max(0.9, min(1.75, float(in_lra) * 0.28)))
        mild_loss = bool(lra_loss is not None and lra_loss > 0.35)
        _out_lufs_for_bamix = _airqc4_analysis_lufs(a)
        _bamix_floor_met = bool(bamix_density_qc and _out_lufs_for_bamix is not None and bamix_floor_lufs is not None and float(_out_lufs_for_bamix) >= float(bamix_floor_lufs) - 0.12)
        _bamix_lra_cost_limit = max(2.2, min(3.1, float(in_lra) * 0.55))
        if bamix_density_qc and _bamix_floor_met and float(out_lra) >= 2.45 and lra_loss is not None and float(lra_loss) <= _bamix_lra_cost_limit:
            lra_class = "bamix_density_completion_lra_cost_within_completion_policy"
            # v63.1.9.4: this governance function may be re-run after an
            # earlier, policy-blind pass already added the old collapse labels.
            # Density-completion treats this as monitored LRA cost, so scrub the
            # stale hard-damage labels in addition to the original absolute-floor
            # warning.
            for _stale_lra_warning in ("lra_too_flat", "mastering_induced_lra_collapse", "lra_loss_relative"):
                if _stale_lra_warning in reasons and _stale_lra_warning not in remove:
                    remove.append(_stale_lra_warning)
            add.append("lra_loss_monitor")
            safe_push_adjustment = "monitor_density_completion_lra_cost_do_not_reject_delivered_target_met_final"
        elif source_flat and low_final and not mild_loss:
            lra_class = "source_lra_already_flat_not_mastering_damage"
            if "lra_too_flat" in reasons:
                remove.append("lra_too_flat")
            add.append("source_lra_already_flat")
            safe_push_adjustment = "cap_future_push_do_not_reject_delivered_final_for_absolute_floor_only"
        elif source_flat and mild_loss:
            lra_class = "source_flat_plus_additional_lra_loss"
            add.append("source_flat_plus_additional_lra_loss")
            safe_push_adjustment = "reduce_future_push"
            candidate_rejection_required = bool(severe_loss)
        elif severe_loss:
            lra_class = "mastering_induced_lra_collapse"
            # Replace the generic absolute-floor warning with the input-relative
            # cause so downstream ledgers do not report both the old raw warning
            # and the more precise adaptive diagnosis as separate issues.
            if "lra_too_flat" in reasons:
                remove.append("lra_too_flat")
            add.append("mastering_induced_lra_collapse")
            add.append("lra_loss_relative")
            candidate_rejection_required = True
            safe_push_adjustment = "reject_louder_candidate_or_rollback"
        elif low_final:
            lra_class = "low_final_lra_with_acceptable_input_relative_loss"
            if "lra_too_flat" in reasons and lra_loss is not None and lra_loss <= 0.35:
                remove.append("lra_too_flat")
                add.append("lra_flat_but_input_relative_loss_small")
            safe_push_adjustment = "monitor_do_not_expand"
        else:
            lra_class = "lra_retained_or_within_adaptive_floor"
    elif out_lra is not None:
        if float(out_lra) < target_min:
            lra_class = "output_lra_low_input_unavailable"
            add.append("lra_low_input_unavailable")

    in_crest = _airqc_num(b.get("crest_factor_db"), None)
    if bamix_qc_policy.get("active"):
        _pm_crest = _airqc_num(bamix_qc_policy.get("premaster_crest_factor_db"), None)
        if _pm_crest is not None:
            in_crest = _pm_crest
            bamix_reference_override["active"] = True
            bamix_reference_override["crest_factor_db"] = round(float(_pm_crest), 3)
    out_crest = _airqc_num(a.get("crest_factor_db"), None)
    crest_loss = (float(in_crest) - float(out_crest)) if in_crest is not None and out_crest is not None else None
    crest_class = "unknown"
    if crest_loss is not None:
        if bamix_density_qc and bamix_max_crest_loss is not None:
            bamix_crest_grace = float(_env_float("BUSY_BAMIX_V6319_ADAPTIVE_QC_CREST_GRACE_DB", 0.18, 0.0, 0.6))
            if crest_loss <= float(bamix_max_crest_loss):
                crest_class = "bamix_density_completion_crest_within_policy"
                if crest_loss > max(1.2, float(bamix_max_crest_loss) * 0.55):
                    add.append("crest_loss_monitor")
            elif crest_loss <= float(bamix_max_crest_loss) + bamix_crest_grace:
                crest_class = "bamix_density_completion_crest_near_policy_boundary"
                add.append("crest_loss_monitor")
                add.append("bamix_crest_boundary_soft_warning")
            else:
                crest_class = "bamix_density_completion_crest_over_policy"
                add.append("crest_loss_relative")
                candidate_rejection_required = True
        elif crest_loss > 2.0:
            crest_class = "mastering_induced_crest_collapse"
            add.append("crest_loss_relative")
            candidate_rejection_required = True
        elif crest_loss > 0.9:
            crest_class = "moderate_crest_loss"
            add.append("crest_loss_monitor")
        else:
            crest_class = "crest_retained"

    in_tp = _analysis_true_peak_authority(b)
    if bamix_qc_policy.get("active"):
        _pm_tp = _airqc_num(bamix_qc_policy.get("premaster_true_peak_dbtp"), None)
        if _pm_tp is not None:
            in_tp = _pm_tp
            bamix_reference_override["active"] = True
            bamix_reference_override["true_peak_dbtp"] = round(float(_pm_tp), 3)
    out_tp = _analysis_true_peak_authority(a)
    tp_pressure_delta = (float(out_tp) - float(in_tp)) if in_tp is not None and out_tp is not None else None
    tp_class = "unknown"
    if out_tp is not None:
        if out_tp > -0.3:
            tp_class = "reject_true_peak_ceiling_violation"
            candidate_rejection_required = True
        elif out_tp > -1.0:
            tp_class = "high_true_peak_pressure"
            add.append("true_peak_pressure_high")
        elif out_tp > -2.0:
            tp_class = "caution_true_peak_pressure"
        else:
            tp_class = "true_peak_safe_margin"

    in_lowmid = _analysis_band_db(b, "low_mid_180_500", "low_mid_db", "mud_index_db")
    out_lowmid = _analysis_band_db(a, "low_mid_180_500", "low_mid_db", "mud_index_db")
    lowmid_delta = (float(out_lowmid) - float(in_lowmid)) if in_lowmid is not None and out_lowmid is not None else None
    in_front = _analysis_band_db(b, "presence_index_db", "vocal_presence_index_db", "presence_2000_5000")
    out_front = _analysis_band_db(a, "presence_index_db", "vocal_presence_index_db", "presence_2000_5000")
    front_delta = (float(out_front) - float(in_front)) if in_front is not None and out_front is not None else None
    if front_delta is not None and front_delta < -0.8:
        add.append("vocal_frontness_relative_loss")
    if lowmid_delta is not None and lowmid_delta > 1.2:
        add.append("low_mid_translation_worsened")

    in_corr = _analysis_ratio(b, "correlation", "corr", default=None)
    out_corr = _analysis_ratio(a, "correlation", "corr", default=None)
    corr_delta = (float(out_corr) - float(in_corr)) if in_corr is not None and out_corr is not None else None
    in_side_high = _analysis_ratio(b, "side_ratio_highband", "side_ratio_high", default=None)
    out_side_high = _analysis_ratio(a, "side_ratio_highband", "side_ratio_high", default=None)
    side_high_delta = (float(out_side_high) - float(in_side_high)) if in_side_high is not None and out_side_high is not None else None
    if corr_delta is not None and corr_delta < -0.25:
        add.append("stereo_correlation_relative_loss")
    if side_high_delta is not None and side_high_delta > 0.18:
        add.append("side_high_smear_relative_increase")

    in_hf = _analysis_band_db(b, "air_12000_18000", "brilliance_7000_12000", "hf_hash_index_db", "harshness_index_db")
    out_hf = _analysis_band_db(a, "air_12000_18000", "brilliance_7000_12000", "hf_hash_index_db", "harshness_index_db")
    hf_delta = (float(out_hf) - float(in_hf)) if in_hf is not None and out_hf is not None else None
    if hf_delta is not None and hf_delta > 1.2:
        add.append("hf_hash_relative_increase")

    damage_scoring = _v62_9_9_4_input_relative_damage_scoring(
        b,
        a,
        decision if isinstance(decision, dict) else {},
        mode,
        lra_loss=lra_loss,
        crest_loss=crest_loss,
        in_lra=in_lra,
        in_crest=in_crest,
        in_tp=in_tp,
        out_tp=out_tp,
    )
    score_decision = damage_scoring.get("decision") if isinstance(damage_scoring.get("decision"), dict) else {}
    score_warning = score_decision.get("recommended_warning")
    if enabled and score_warning:
        add.append(str(score_warning))
    score_hard_veto = bool(score_decision.get("hard_veto")) if isinstance(score_decision, dict) else False

    adjusted = sorted((set(reasons) - set(remove)) | set(add)) if enabled else sorted(set(reasons))
    return {
        "schema_version": "busy_adaptive_input_relative_qc_governance_v8_5_3_62_9_9_4_4",
        "enabled": enabled,
        "active": enabled,
        "policy": "Final fixed thresholds remain safety references, but warnings and candidate policy are reclassified by input-to-output loss/retention so originally-flat sources are not mistaken for mastering damage.",
        "bamix_effective_input_reference": {**bamix_reference_override, "source": "bamix_premaster_handoff_policy" if bamix_reference_override.get("active") else "analysis_before_snapshot", "handoff_mode": bamix_qc_mode or None, "max_crest_loss_db": round(float(bamix_max_crest_loss), 3) if bamix_max_crest_loss is not None else None},
        "lra": {
            "input_lra_lu": round(float(in_lra), 3) if in_lra is not None else None,
            "final_lra_lu": round(float(out_lra), 3) if out_lra is not None else None,
            "lra_loss_lu": round(float(lra_loss), 3) if lra_loss is not None else None,
            "lra_retention_ratio": round(float(lra_retention), 4) if lra_retention is not None else None,
            "absolute_reference_floor_lu": round(float(target_min), 3),
            "adaptive_lra_floor_lu": round(float(adaptive_floor), 3) if adaptive_floor is not None else None,
            "classification": lra_class,
            "safe_push_budget_adjustment": safe_push_adjustment,
        },
        "crest": {
            "input_crest_db": round(float(in_crest), 3) if in_crest is not None else None,
            "final_crest_db": round(float(out_crest), 3) if out_crest is not None else None,
            "crest_loss_db": round(float(crest_loss), 3) if crest_loss is not None else None,
            "classification": crest_class,
        },
        "true_peak_pressure": {
            "input_true_peak_dbtp": round(float(in_tp), 3) if in_tp is not None else None,
            "final_true_peak_dbtp": round(float(out_tp), 3) if out_tp is not None else None,
            "true_peak_pressure_delta_db": round(float(tp_pressure_delta), 3) if tp_pressure_delta is not None else None,
            "classification": tp_class,
        },
        "low_end_frontness": {
            "low_mid_delta_db": round(float(lowmid_delta), 3) if lowmid_delta is not None else None,
            "frontness_delta_db": round(float(front_delta), 3) if front_delta is not None else None,
        },
        "stereo_hf_codec": {
            "correlation_delta": round(float(corr_delta), 4) if corr_delta is not None else None,
            "side_high_delta": round(float(side_high_delta), 4) if side_high_delta is not None else None,
            "hf_delta_db": round(float(hf_delta), 3) if hf_delta is not None else None,
        },
        "warning_reclassification": {
            "original_warnings": sorted(set(reasons)),
            "removed_warnings": sorted(set(remove)),
            "added_warnings": sorted(set(add)),
            "adjusted_warnings": adjusted,
        },
        "updated_warnings": adjusted,
        "input_relative_damage_scoring": damage_scoring,
        "adaptive_input_relative_qc_score": {
            "schema_version": "busy_adaptive_input_relative_qc_score_alias_v8_5_3_62_9_9_4_4",
            "weighted_risk": damage_scoring.get("weighted_risk") if isinstance(damage_scoring, dict) else None,
            "decision": damage_scoring.get("decision") if isinstance(damage_scoring, dict) else {},
            "axis_results": damage_scoring.get("axis_results") if isinstance(damage_scoring, dict) else [],
        },
        "legacy_candidate_rejection_signal_by_relative_damage": bool(candidate_rejection_required),
        "candidate_rejection_required_by_relative_damage": bool(score_hard_veto),
    }



def _v62_9_9_3_2_reconcile_adaptive_input_relative_qc_report(report: dict[str, Any] | None, decision: dict[str, Any] | None = None, mode: str = "") -> dict[str, Any]:
    """v62.9.9.4: adaptive QC measurement-authority + damage-scoring reconciliation.

    The adaptive input-relative QC must compare the original input against the
    delivered final master. Candidate telemetry such as
    commercial_loudness_finish.output can describe an attempted louder master,
    but it is not the delivered final authority unless the selected/exported
    buffer actually came from that candidate. This helper is report-only and
    performs no DSP. It re-runs the adaptive QC block from report-level
    authoritative sources after final metric aliases have been normalized.
    """
    if not isinstance(report, dict):
        return report or {}

    def _dict_at(obj: Any, *path: str) -> dict[str, Any]:
        cur = obj
        for key in path:
            if not isinstance(cur, dict):
                return {}
            cur = cur.get(key)
        return cur if isinstance(cur, dict) else {}

    def _first_dict_with_metric(candidates: list[tuple[str, dict[str, Any]]], metric: str) -> tuple[str, dict[str, Any]]:
        for label, cand in candidates:
            if not isinstance(cand, dict) or not cand:
                continue
            try:
                if metric == "lra" and _analysis_lra_lu(cand) is not None:
                    return label, cand
                if metric == "tp" and _analysis_true_peak_authority(cand) is not None:
                    return label, cand
            except Exception:
                continue
        for label, cand in candidates:
            if isinstance(cand, dict) and cand:
                return label, cand
        return "unavailable", {}

    pre = report.get("pre_limiter_report") if isinstance(report.get("pre_limiter_report"), dict) else {}
    inp = report.get("input_analysis") if isinstance(report.get("input_analysis"), dict) else {}
    original = report.get("original_input_analysis_before_pre_clean") if isinstance(report.get("original_input_analysis_before_pre_clean"), dict) else {}
    dec = decision if isinstance(decision, dict) else report.get("decision") if isinstance(report.get("decision"), dict) else {}

    input_candidates = [
        ("report.original_input_analysis_before_pre_clean", original),
        ("report.input_analysis.original_input_analysis_before_pre_clean", _dict_at(inp, "original_input_analysis_before_pre_clean")),
        ("report.pre_limiter_report.original_input_analysis_before_pre_clean", _dict_at(pre, "original_input_analysis_before_pre_clean")),
        ("report.pre_limiter_report.input_analysis.original_input_analysis_before_pre_clean", _dict_at(pre, "input_analysis", "original_input_analysis_before_pre_clean")),
        ("report.input_analysis", inp),
        ("report.pre_limiter_report.input_analysis", _dict_at(pre, "input_analysis")),
        ("decision.v62_9_9_research_governance.microdynamics_lra_shape_recovery", _dict_at(dec, "v62_9_9_research_governance", "microdynamics_lra_shape_recovery")),
    ]
    input_label, input_source = _first_dict_with_metric(input_candidates, "lra")

    output = report.get("output_analysis") if isinstance(report.get("output_analysis"), dict) else {}
    final_alias_source = {
        "lra_lu": report.get("lra_lu"),
        "crest_factor_db": report.get("crest_factor_db"),
        "true_peak_dbtp": report.get("true_peak_dbtp"),
        "true_peak_dbfs": report.get("true_peak_dbtp"),
        "approx_true_peak_dbfs": report.get("true_peak_dbtp"),
        "integrated_lufs": report.get("final_lufs"),
        "stereo_integrated_lufs": report.get("final_lufs_stereo"),
        "correlation": (output or {}).get("correlation"),
        "quality_indices": (output or {}).get("quality_indices", {}) if isinstance(output, dict) else {},
        "ms": (output or {}).get("ms", {}) if isinstance(output, dict) else {},
        "bands_db": (output or {}).get("bands_db", {}) if isinstance(output, dict) else {},
    }
    final_label, final_source = _first_dict_with_metric([
        ("report.output_analysis_delivered_final", output),
        ("report.top_level_final_aliases", final_alias_source),
    ], "tp")
    if isinstance(output, dict) and output:
        tp = _analysis_true_peak_authority(output)
        if tp is None:
            tp = _airqc_num(report.get("true_peak_dbtp"), None)
        if tp is not None:
            tp_r = round(float(tp), 3)
            for k in ("true_peak_dbtp", "true_peak_dbfs", "approx_true_peak_dbfs"):
                output[k] = tp_r
            loud = output.get("loudness") if isinstance(output.get("loudness"), dict) else {}
            for k in ("true_peak_dbtp", "true_peak_dbfs", "approx_true_peak_dbfs"):
                loud[k] = tp_r
            output["loudness"] = loud
            final_source = output
            final_label = "report.output_analysis_delivered_final"

    lra_guard = report.get("lra_guard") if isinstance(report.get("lra_guard"), dict) else {}
    old_airqc = report.get("adaptive_input_relative_qc_governance") if isinstance(report.get("adaptive_input_relative_qc_governance"), dict) else {}
    old_wr = old_airqc.get("warning_reclassification") if isinstance(old_airqc.get("warning_reclassification"), dict) else {}
    if isinstance(old_wr, dict) and old_wr.get("original_warnings"):
        raw_warnings = [str(x) for x in old_wr.get("original_warnings", [])]
    elif isinstance(lra_guard, dict) and lra_guard.get("raw_warnings_before_adaptive_input_relative_qc"):
        raw_warnings = [str(x) for x in lra_guard.get("raw_warnings_before_adaptive_input_relative_qc", [])]
    else:
        raw_warnings = [str(x) for x in report.get("remaining_safety_warnings", []) or []]

    def _is_redacted_adaptive_marker(_x: Any) -> bool:
        _s = str(_x)
        return (
            _s.startswith("legacy_warning_snapshot_redacted_after_adaptive_qc")
            or _s == "adaptive_qc_legacy_value_reclassified"
            or _s == "adaptive_qc_warning_reclassified"
        )

    raw_warnings = [str(x) for x in raw_warnings if not _is_redacted_adaptive_marker(x)]

    new_airqc = _adaptive_input_relative_qc_governance(
        input_source,
        final_source,
        dec,
        mode,
        lra_guard,
        raw_warnings,
    )
    new_airqc["schema_version"] = "busy_adaptive_input_relative_qc_governance_v8_5_3_62_9_9_4_4"
    _bamix_ref_for_authority = new_airqc.get("bamix_effective_input_reference") if isinstance(new_airqc.get("bamix_effective_input_reference"), dict) else {}
    if _bamix_ref_for_authority.get("active"):
        input_label = "decision.busy_auto_mixing_mastering_handoff_policy.bamix_premaster_metrics"
    new_airqc["measurement_authority"] = {
        "schema_version": "busy_v62_9_9_4_4_adaptive_qc_deep_lineage_handoff_scrub_hotfix",
        "input_source": input_label,
        "final_source": final_label,
        "delivered_final_keys_preferred": ["output_analysis", "top_level_final_aliases"],
        "candidate_metric_exclusion": "commercial_loudness_finish.output is candidate/commercial-attempt telemetry and is not used as delivered final authority unless selected/export-locked as final.",
    }
    clf = report.get("commercial_loudness_finish") if isinstance(report.get("commercial_loudness_finish"), dict) else {}
    cand_out = clf.get("output") if isinstance(clf.get("output"), dict) else {}
    if cand_out:
        new_airqc["candidate_reference_not_delivered_final"] = {
            "source": "commercial_loudness_finish.output",
            "selected": clf.get("selected"),
            "reason": clf.get("reason"),
            "candidate_true_peak_dbtp": _analysis_true_peak_authority(cand_out),
            "candidate_lra_lu": _analysis_lra_lu(cand_out),
            "policy": "candidate metrics can inform candidate risk, not delivered-final warnings.",
        }

    old_adjusted = set(str(x) for x in (old_wr.get("adjusted_warnings", []) if isinstance(old_wr, dict) else []) if not _is_redacted_adaptive_marker(x))
    current = set(str(x) for x in (report.get("remaining_safety_warnings", []) or []) if not _is_redacted_adaptive_marker(x))
    unrelated = current - old_adjusted if old_adjusted else set()
    new_wr = new_airqc.get("warning_reclassification") if isinstance(new_airqc.get("warning_reclassification"), dict) else {}
    adjusted = set(str(x) for x in (new_wr.get("adjusted_warnings", []) if isinstance(new_wr, dict) else []) if not _is_redacted_adaptive_marker(x))
    final_warnings = sorted(x for x in (unrelated | adjusted) if not _is_redacted_adaptive_marker(x))

    report["adaptive_input_relative_qc_governance"] = new_airqc
    report["remaining_safety_warnings"] = final_warnings
    report["engine_version"] = V6362_MASTERING_ENGINE_VERSION
    report["pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_63_3_4_1_aug_telemetry_reason_cleanup"
    report["mastering_report_schema_version"] = "busy_master_report_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"

    # v62.9.9.4.3: surface input metric aliases at top level so debug
    # brief/Supabase readers do not have to know every historical nested
    # source path.  These aliases are report-only and come from the same
    # input authority used by Adaptive QC.
    _input_lufs = _airqc4_analysis_lufs(input_source)
    _input_lra = _analysis_lra_lu(input_source)
    _input_tp = _analysis_true_peak_authority(input_source)
    _input_crest = _airqc_num(input_source.get("crest_factor_db"), None) if isinstance(input_source, dict) else None
    _bamix_ref_alias = new_airqc.get("bamix_effective_input_reference") if isinstance(new_airqc.get("bamix_effective_input_reference"), dict) else {}
    if _bamix_ref_alias.get("active"):
        _input_lufs = _airqc_num(_bamix_ref_alias.get("integrated_lufs"), _input_lufs)
        _input_lra = _airqc_num(_bamix_ref_alias.get("lra_lu"), _input_lra)
        _input_tp = _airqc_num(_bamix_ref_alias.get("true_peak_dbtp"), _input_tp)
        _input_crest = _airqc_num(_bamix_ref_alias.get("crest_factor_db"), _input_crest)
    if _input_lufs is not None:
        report["input_lufs"] = round(float(_input_lufs), 3)
    if _input_lra is not None:
        report["input_lra_lu"] = round(float(_input_lra), 3)
    if _input_tp is not None:
        report["input_true_peak_dbtp"] = round(float(_input_tp), 3)
    if _input_crest is not None:
        report["input_crest_factor_db"] = round(float(_input_crest), 3)
    report["input_metric_authority_v62_9_9_4_4"] = {
        "schema_version": "busy_input_metric_authority_v8_5_3_62_9_9_4_4",
        "source": input_label,
        "policy": "Top-level input aliases mirror the same input source used by Adaptive Input-Relative QC.",
    }

    if isinstance(lra_guard, dict):
        lra_payload = new_airqc.get("lra") if isinstance(new_airqc.get("lra"), dict) else {}
        in_lra = lra_payload.get("input_lra_lu")
        out_lra = lra_payload.get("final_lra_lu")
        if in_lra is not None:
            lra_guard["input_lra_lu"] = in_lra
        if out_lra is not None:
            lra_guard["output_lra_lu"] = out_lra
        if in_lra is not None and out_lra is not None:
            try:
                lra_guard["lra_delta_lu"] = round(float(out_lra) - float(in_lra), 3)
            except Exception:
                pass
        if not lra_guard.get("raw_warnings_before_adaptive_input_relative_qc"):
            lra_guard["raw_warnings_before_adaptive_input_relative_qc"] = list(raw_warnings)
        lra_guard["adaptive_input_relative_reclassification"] = new_wr
        lra_added = [
            str(x) for x in (new_wr.get("added_warnings", []) if isinstance(new_wr, dict) else [])
            if str(x) in {
                "source_lra_already_flat",
                "lra_flat_but_input_relative_loss_small",
                "source_flat_plus_additional_lra_loss",
                "mastering_induced_lra_collapse",
                "lra_low_input_unavailable",
            }
        ]
        lra_removed = set(str(x) for x in (new_wr.get("removed_warnings", []) if isinstance(new_wr, dict) else []))
        raw_lra_display = [str(x) for x in (lra_guard.get("raw_warnings_before_adaptive_input_relative_qc") or [])]
        lra_display = [x for x in raw_lra_display if x not in lra_removed]
        for w in lra_added:
            if w not in lra_display:
                lra_display.append(w)
        lra_guard["warnings"] = sorted(lra_display)
        lra_guard["adaptive_adjusted_lra_warnings"] = sorted(lra_display)
        report["lra_guard"] = lra_guard

    # Keep downstream debug/ledger views synchronized with the adaptive QC
    # warning authority.  v62.9.9.4.3 deepens the handoff scrub: stale legacy
    # warnings can be embedded in warning_details, events, final_reason strings,
    # v57 delivered-final gates, Gemini lineage context, and private_internal
    # debug feature copies.  Those fields are report/lineage only; the delivered
    # audio and DSP are not changed here.
    _legacy_warning_codes = {
        "lra_low_input_unavailable",
        "lra_too_flat",
        "true_peak_pressure_high",
        "lra_unknown_conservative_budget",
    }
    _legacy_scrub_counts: dict[str, int] = {}

    def _bump_scrub(_what: str) -> None:
        _legacy_scrub_counts[_what] = int(_legacy_scrub_counts.get(_what, 0)) + 1

    def _contains_legacy_warning(_value: Any) -> bool:
        if isinstance(_value, str):
            return any(code in _value for code in _legacy_warning_codes)
        if isinstance(_value, list):
            return any(_contains_legacy_warning(v) for v in _value)
        if isinstance(_value, dict):
            return any(_contains_legacy_warning(v) for v in _value.values())
        return False

    def _adaptive_final_reason() -> str:
        _fl = _airqc4_analysis_lufs(final_source)
        _tp = _analysis_true_peak_authority(final_source)
        _cr = _airqc_num(final_source.get("crest_factor_db"), None) if isinstance(final_source, dict) else None
        _parts = []
        if _fl is not None:
            _parts.append(f"{float(_fl):.2f} LUFS")
        if _tp is not None:
            _parts.append(f"{float(_tp):.2f} dBTP")
        if _cr is not None:
            _parts.append(f"crest {float(_cr):.2f} dB")
        _metric_text = ", ".join(_parts) if _parts else "delivered final metrics available"
        _warn_text = ", ".join(str(x) for x in final_warnings) if final_warnings else "none"
        return f"Final master measured {_metric_text}. Adaptive input-relative QC final warnings: {_warn_text}."

    def _adaptive_warning_details() -> list[dict[str, Any]]:
        _details: list[dict[str, Any]] = []
        for _code in final_warnings:
            _details.append({
                "code": str(_code),
                "source": "adaptive_input_relative_qc_governance.updated_warnings",
                "severity": "warning" if str(_code) != "adaptive_qc_compare_softer_candidate" else "caution",
            })
        return _details

    def _retry_blocked_by_replacement() -> list[str]:
        if "adaptive_qc_compare_softer_candidate" in final_warnings:
            return ["adaptive_qc_compare_softer_candidate"]
        return list(final_warnings)

    def _redacted_legacy_snapshot() -> list[str]:
        return ["legacy_warning_snapshot_redacted_after_adaptive_qc_v62_9_9_4_4"]

    def _sync_warning_count_metrics(_obj: dict[str, Any], _scope: str = "") -> None:
        """v62.9.9.4.4: align report-only warning_count fields with the
        post-adaptive warning authority.

        This intentionally does not recompute confidence/audio decisions; it
        only prevents nested metric summaries from showing stale pre-adaptive
        warning counts after Adaptive QC has replaced the active warnings.
        """
        if not isinstance(_obj, dict):
            return
        _metrics = _obj.get("metrics")
        if isinstance(_metrics, dict) and "warning_count" in _metrics:
            try:
                _old_wc = int(_metrics.get("warning_count") or 0)
            except Exception:
                _old_wc = _metrics.get("warning_count")
            _new_wc = len(list(final_warnings))
            if _metrics.get("warning_count") != _new_wc:
                _metrics.setdefault("warning_count_before_adaptive_qc", _old_wc)
                _metrics["warning_count"] = _new_wc
                _metrics["warning_count_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.updated_warnings"
                _metrics["warning_count_sync_note"] = "Report-only handoff sync; existing confidence score is not recomputed by this hotfix."
                _bump_scrub("warning_count_synced")
            _obj["metrics"] = _metrics

    def _sync_warning_container(_obj: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(_obj, dict):
            return _obj
        if _obj.get("remaining_safety_warnings") and not _obj.get("raw_remaining_safety_warnings_before_adaptive_qc"):
            _obj["raw_remaining_safety_warnings_before_adaptive_qc_count"] = len(list(_obj.get("remaining_safety_warnings") or []))
        if _obj.get("warnings") and not _obj.get("raw_warnings_before_adaptive_qc"):
            _obj["raw_warnings_before_adaptive_qc_count"] = len(list(_obj.get("warnings") or []))
        if _obj.get("reason_codes") and not _obj.get("raw_reason_codes_before_adaptive_qc"):
            _obj["raw_reason_codes_before_adaptive_qc_count"] = len(list(_obj.get("reason_codes") or []))
        _obj["remaining_safety_warnings"] = list(final_warnings)
        _obj["adaptive_qc_final_warnings"] = list(final_warnings)
        _obj["adaptive_qc_final_reason_codes"] = list(final_warnings)
        _obj["adaptive_input_relative_damage_scoring"] = new_airqc.get("input_relative_damage_scoring", {})
        if "reason_codes" in _obj:
            _obj["reason_codes"] = list(final_warnings)
        _sync_warning_count_metrics(_obj, "targeted_warning_container")
        if isinstance(_obj.get("warning_details"), list):
            _obj["warning_details"] = _adaptive_warning_details()
        if isinstance(_obj.get("final_reason"), str) and _contains_legacy_warning(_obj.get("final_reason")):
            _obj["raw_final_reason_before_adaptive_qc_redacted"] = True
            _obj["final_reason"] = _adaptive_final_reason()
        rb = _obj.get("rollback")
        if isinstance(rb, dict):
            if rb.get("reason_codes") and not rb.get("raw_reason_codes_before_adaptive_qc"):
                rb["raw_reason_codes_before_adaptive_qc_count"] = len(list(rb.get("reason_codes") or []))
            rb["reason_codes"] = list(final_warnings)
            rb["adaptive_qc_final_reason_codes"] = list(final_warnings)
            rb["reason_code_authority"] = "adaptive_input_relative_qc_governance.updated_warnings"
            _obj["rollback"] = rb
        return _obj

    def _backfill_v57_lra_fields(_value: dict[str, Any], _path: tuple[str, ...]) -> None:
        """v62.9.9.4.4: fill remaining nested v57/delivered gate LRA
        telemetry from Adaptive QC authority.  This is report-only lineage
        hygiene; it does not change candidate selection or DSP.
        """
        if not isinstance(_value, dict):
            return
        _path_text = ".".join(str(x) for x in _path)
        _lra_payload = new_airqc.get("lra") if isinstance(new_airqc.get("lra"), dict) else {}
        _in_lra = _airqc_num(_lra_payload.get("input_lra_lu"), None)
        _out_lra = _airqc_num(_lra_payload.get("final_lra_lu"), None)
        if _in_lra is None and report.get("input_lra_lu") is not None:
            _in_lra = _airqc_num(report.get("input_lra_lu"), None)
        if _out_lra is None and report.get("lra_lu") is not None:
            _out_lra = _airqc_num(report.get("lra_lu"), None)
        if _in_lra is None and _out_lra is None:
            return
        _looks_v57_lra = (
            "v57" in _path_text
            or "delivered_final_gate" in _path_text
            or "delivered_final_v57_damage_gate" in _path_text
            or "v57_2_lra_resolution" in _path_text
            or "lra_resolution" in _path_text
        )
        _has_lra_keys = any(k in _value for k in (
            "input_lra_lu", "output_lra_lu", "final_lra_lu", "candidate_lra_lu",
            "lra_delta_lu", "lra_delta_vs_input_lu", "input_lra_source", "output_lra_source",
        ))
        if not (_looks_v57_lra and _has_lra_keys):
            return
        _changed = False
        if _value.get("input_lra_lu") is None and _in_lra is not None:
            _value["input_lra_lu"] = round(float(_in_lra), 3)
            _value["input_lra_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.lra.input_lra_lu"
            _value["input_lra_source"] = _value.get("input_lra_source") or "adaptive_input_relative_qc_governance"
            _bump_scrub("nested_input_lra_backfilled")
            _changed = True
        if _value.get("output_lra_lu") is None and _out_lra is not None:
            _value["output_lra_lu"] = round(float(_out_lra), 3)
            _value["output_lra_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.lra.final_lra_lu"
            _value["output_lra_source"] = _value.get("output_lra_source") or "adaptive_input_relative_qc_governance"
            _bump_scrub("nested_output_lra_backfilled")
            _changed = True
        _candidate_lra = _airqc_num(_value.get("candidate_lra_lu"), None)
        if _candidate_lra is None:
            _candidate_lra = _airqc_num(_value.get("output_lra_lu"), None) or _out_lra
        if _value.get("lra_delta_vs_input_lu") is None and _candidate_lra is not None and _in_lra is not None:
            _value["lra_delta_vs_input_lu"] = round(float(_candidate_lra) - float(_in_lra), 3)
            _value["lra_delta_vs_input_authority_v62_9_9_4_4"] = "candidate/output LRA minus adaptive input LRA"
            _bump_scrub("nested_lra_delta_vs_input_backfilled")
            _changed = True
        if _value.get("lra_delta_lu") is None:
            _calc_out = _airqc_num(_value.get("output_lra_lu"), None) or _out_lra
            _calc_in = _airqc_num(_value.get("input_lra_lu"), None) or _in_lra
            if _calc_out is not None and _calc_in is not None:
                _value["lra_delta_lu"] = round(float(_calc_out) - float(_calc_in), 3)
                _value["lra_delta_authority_v62_9_9_4_4"] = "output LRA minus input LRA from adaptive QC authority"
                _bump_scrub("nested_lra_delta_backfilled")
                _changed = True
        if _changed:
            _value["lra_lineage_backfill_v62_9_9_4_4"] = True

    def _scrub_legacy_warning_lineage(_value: Any, _path: tuple[str, ...] = ()) -> Any:
        _key = _path[-1] if _path else ""
        _path_text = ".".join(_path)
        # v63.1.4.1: the v63.1.3.3 LRA semantic split is already explicit
        # telemetry, not an active warning container.  The legacy deep scrub may
        # still recurse into it and replace semantic buckets such as
        # absolute_threshold_warnings=[lra_too_flat] with the full final warning
        # list.  That collapses the very distinction this marker was created to
        # expose.  Preserve this subtree verbatim; active warnings remain governed
        # by remaining_safety_warnings / adaptive_input_relative_qc_governance.
        if "warning_semantics_v6313_3" in _path_text:
            return _value
        if isinstance(_value, dict):
            _sync_warning_count_metrics(_value, ".".join(_path))
            _backfill_v57_lra_fields(_value, _path)
            # v57 feasibility blocks were created before the input-authority
            # handoff, so their input/output LRA can remain null even when the
            # report now has authoritative input/final LRA.  Backfill only the
            # report fields and mark the source; do not change DSP policy.
            if _key == "v57_commercial_loudness_feasibility":
                _lra_payload = new_airqc.get("lra") if isinstance(new_airqc.get("lra"), dict) else {}
                if _value.get("input_lra_lu") is None and _lra_payload.get("input_lra_lu") is not None:
                    _value["input_lra_lu"] = _lra_payload.get("input_lra_lu")
                    _value["input_lra_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.lra.input_lra_lu"
                    _bump_scrub("v57_input_lra_backfilled")
                if _value.get("output_lra_lu") is None and _lra_payload.get("final_lra_lu") is not None:
                    _value["output_lra_lu"] = _lra_payload.get("final_lra_lu")
                    _value["output_lra_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.lra.final_lra_lu"
                    _bump_scrub("v57_output_lra_backfilled")
                if _contains_legacy_warning(_value.get("retry_blocked_by")):
                    _value["retry_blocked_by_before_adaptive_qc_redacted"] = True
                    _value["retry_blocked_by"] = _retry_blocked_by_replacement()
                    _bump_scrub("v57_retry_blocked_by_replaced")
            for _k in list(_value.keys()):
                _v = _value.get(_k)
                _lower = str(_k).lower()
                if _k in {"warning_details"} and isinstance(_v, list):
                    if _contains_legacy_warning(_v):
                        _value[_k] = _adaptive_warning_details()
                        _bump_scrub("warning_details_replaced")
                    continue
                if _k in {"warnings", "warnings_considered", "remaining_safety_warnings", "reason_codes", "adaptive_qc_final_warnings", "adaptive_qc_final_reason_codes"} and isinstance(_v, list):
                    if _contains_legacy_warning(_v) or _k in {"warnings_considered", "reason_codes", "remaining_safety_warnings"}:
                        _value[_k] = list(final_warnings)
                        _value[f"{_k}_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.updated_warnings"
                        _bump_scrub(f"{_k}_replaced")
                    continue
                if _k in {"retry_blocked_by", "commercial_retry_blocked_by"} and isinstance(_v, list):
                    if _contains_legacy_warning(_v):
                        _value[f"{_k}_before_adaptive_qc_redacted"] = True
                        _value[_k] = _retry_blocked_by_replacement()
                        _bump_scrub(f"{_k}_replaced")
                    continue
                if _k in {"original_warnings", "removed_warnings", "raw_warnings_before_adaptive_qc", "raw_warnings_before_adaptive_input_relative_qc", "raw_reason_codes_before_adaptive_qc", "raw_remaining_safety_warnings_before_adaptive_qc"} and isinstance(_v, list):
                    if _contains_legacy_warning(_v):
                        _value[f"{_k}_count_before_redaction"] = len(_v)
                        _value[_k] = _redacted_legacy_snapshot()
                        _bump_scrub("raw_legacy_snapshot_redacted")
                    continue
                if isinstance(_v, str) and _contains_legacy_warning(_v):
                    if _k == "final_reason":
                        _value["raw_final_reason_before_adaptive_qc_redacted"] = True
                        _value[_k] = _adaptive_final_reason()
                        _bump_scrub("final_reason_rewritten")
                    elif _k == "code":
                        _value[_k] = "adaptive_qc_warning_reclassified"
                        _value["code_authority_v62_9_9_4_4"] = "adaptive_input_relative_qc_governance.updated_warnings"
                        _bump_scrub("warning_code_reclassified")
                    else:
                        _value[f"{_k}_before_adaptive_qc_redacted"] = True
                        _value[_k] = "adaptive_qc_legacy_value_reclassified"
                        _bump_scrub("string_reclassified")
                    continue
                _value[_k] = _scrub_legacy_warning_lineage(_v, _path + (str(_k),))
            return _value
        if isinstance(_value, list):
            if _contains_legacy_warning(_value):
                if any(str(part).startswith("raw_") or part in {"original_warnings", "removed_warnings"} for part in _path):
                    _bump_scrub("raw_list_redacted")
                    return _redacted_legacy_snapshot()
                # Homogeneous string warning/reason lists should be replaced by
                # the latest adaptive authority. Mixed event arrays are recursed.
                if all(isinstance(x, str) for x in _value):
                    _bump_scrub("string_list_replaced")
                    if _key in {"retry_blocked_by", "commercial_retry_blocked_by"}:
                        return _retry_blocked_by_replacement()
                    return list(final_warnings)
            return [_scrub_legacy_warning_lineage(v, _path + ("[]",)) for v in _value]
        if isinstance(_value, str) and _contains_legacy_warning(_value):
            _bump_scrub("loose_string_reclassified")
            return "adaptive_qc_legacy_value_reclassified"
        return _value

    _v6313_3_lra_semantic_snapshot = None
    try:
        if isinstance(report.get("lra_guard"), dict) and isinstance(report["lra_guard"].get("warning_semantics_v6313_3"), dict):
            _v6313_3_lra_semantic_snapshot = copy.deepcopy(report["lra_guard"].get("warning_semantics_v6313_3"))
    except Exception:
        _v6313_3_lra_semantic_snapshot = None

    for _k in ("post_master_qc_explainer", "rollback_ledger", "final_perceptual_ab_arbiter", "playback_translation_auditor"):
        _obj = report.get(_k)
        if isinstance(_obj, dict):
            report[_k] = _sync_warning_container(_obj)

    # Deep-scrub all active nested copies after the targeted sync pass.  This
    # intentionally also redacts raw legacy-warning snapshots from report-level
    # JSON so grep/search no longer reports stale warnings as current facts.
    report = _scrub_legacy_warning_lineage(report, ("report",))

    def _restore_v6313_3_lra_semantics(_obj: Any) -> None:
        if _v6313_3_lra_semantic_snapshot is None:
            return
        if isinstance(_obj, dict):
            if "warning_semantics_v6313_3" in _obj:
                _obj["warning_semantics_v6313_3"] = copy.deepcopy(_v6313_3_lra_semantic_snapshot)
            for _vv in _obj.values():
                _restore_v6313_3_lra_semantics(_vv)
        elif isinstance(_obj, list):
            for _vv in _obj:
                _restore_v6313_3_lra_semantics(_vv)

    _restore_v6313_3_lra_semantics(report)
    if isinstance(_v6313_3_lra_semantic_snapshot, dict):
        # Keep the top-level alias in sync too; it does not contain the
        # nested warning_semantics_v6313_3 key, so the recursive restore above
        # cannot repair it by itself.
        report["lra_warning_semantics"] = copy.deepcopy(_v6313_3_lra_semantic_snapshot)
    report["v6314_2_lra_semantic_scrub_guard"] = {
        "schema_version": "busy_lra_semantic_scrub_guard_v8_5_3_63_1_4_2",
        "active": bool(_v6313_3_lra_semantic_snapshot is not None),
        "top_level_alias_restored": bool(isinstance(_v6313_3_lra_semantic_snapshot, dict)),
        "policy": "Preserve v63.1.3.3 LRA semantic buckets across legacy warning deep-scrub, including the top-level lra_warning_semantics alias; semantic buckets are telemetry, not active final warning authority.",
    }
    # Backward-compatible marker for dashboards already looking for the v63.1.4.1 key.
    report["v6314_1_lra_semantic_scrub_guard"] = dict(report["v6314_2_lra_semantic_scrub_guard"])
    report["legacy_warning_deep_handoff_scrub_v62_9_9_4_4"] = {
        "schema_version": "busy_legacy_warning_deep_handoff_scrub_v8_5_3_62_9_9_4_4",
        "active": True,
        "policy": "Stale pre-adaptive warnings are removed from active lineage/event/reason fields and redacted from raw snapshots; authoritative current warnings are adaptive_input_relative_qc_governance.updated_warnings.",
        "scrub_counts": dict(sorted(_legacy_scrub_counts.items())),
        "final_warning_authority": "adaptive_input_relative_qc_governance.updated_warnings",
        "final_warnings": list(final_warnings),
    }

    # Preserve the historical alias and add version-specific aliases.
    report["adaptive_qc_measurement_authority_v62_9_9_4"] = new_airqc.get("measurement_authority", {})
    report["adaptive_qc_measurement_authority_v62_9_9_4_2"] = new_airqc.get("measurement_authority", {})
    report["adaptive_qc_measurement_authority_v62_9_9_4_4"] = new_airqc.get("measurement_authority", {})
    return report


def _maybe_apply_lra_recovery_stage(
    final: np.ndarray,
    sr: int,
    before: dict[str, Any],
    current_analysis: dict[str, Any] | None,
    decision: dict[str, Any],
    mode: str,
    commercial_report: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """v8.5.3.47 conservative LRA recovery for already-flat Suno masters.

    This stage is deliberately after candidate selection and before dither/final
    analysis. It is not a new loudness push. It only tests a tiny transient/LRA
    recovery candidate and keeps it if deterministic QC does not get worse.
    """
    report: dict[str, Any] = {
        "schema_version": "busy_lra_recovery_stage_v8_5_3_47_1",
        "enabled": bool(_env_bool("BUSY_V47_LRA_RECOVERY", True)),
        "active": False,
        "applied": False,
        "policy": "flat-input recovery is conservative and bounded; it cannot bypass true-peak, safety, LRA or A/B delta guards",
    }
    if not report["enabled"]:
        report["reason"] = "BUSY_V47_LRA_RECOVERY disabled"
        return final, report
    try:
        cur = current_analysis if isinstance(current_analysis, dict) and current_analysis else analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
        in_lra = _analysis_lra_lu(before)
        out_lra = _analysis_lra_lu(cur)
        if out_lra is None:
            # v8.5.3.62.9.9.2: use the same bounded runtime-safe LRA estimator
            # that final analysis already trusts before falling back to an
            # optional full analyze_audio() call.  This keeps Cloud Run runtime
            # bounded while preventing B109 from reporting LRA unavailable when
            # the delivered final can be measured later.
            try:
                lra_summary = _runtime_safe_lra_summary(final, sr)
                if isinstance(lra_summary, dict) and lra_summary.get("lra_lu") is not None:
                    cur.setdefault("loudness", {})["short_term_lufs"] = lra_summary
                    cur["lra_lu"] = lra_summary.get("lra_lu")
                    out_lra = _analysis_lra_lu(cur)
                    report["runtime_safe_lra_used_for_trigger"] = True
                    report["runtime_safe_lra_summary"] = lra_summary
            except Exception as lra_safe_exc:
                report["runtime_safe_lra_trigger_error"] = str(lra_safe_exc)[:300]
        if out_lra is None:
            # v8.5.3.62.8.2: in v48 candidate-generator mode this stage will not
            # apply a destructive final LRA post-process anyway.  Avoid a late
            # full analyze_audio() call on the delivered buffer unless explicitly
            # requested, because that path can dominate Cloud Run wall time.
            if _env_bool("BUSY_V47_LRA_RECOVERY_FULL_ANALYSIS", False):
                try:
                    cur_full = analyze_audio(final, sr)
                    if isinstance(cur_full, dict):
                        cur = {**cur, **cur_full}
                        out_lra = _analysis_lra_lu(cur)
                        report["full_analysis_used_for_lra_trigger"] = True
                except Exception as lra_exc:
                    report["full_analysis_for_lra_trigger_error"] = str(lra_exc)[:300]
            else:
                report["full_analysis_skipped_for_runtime"] = True
        in_lra_f = float(in_lra) if in_lra is not None else None
        out_lra_f = float(out_lra) if out_lra is not None else None
        report.update({
            "input_lra_lu": round(in_lra_f, 3) if in_lra_f is not None else None,
            "pre_recovery_lra_lu": round(out_lra_f, 3) if out_lra_f is not None else None,
            "pre_recovery_lufs": cur.get("integrated_lufs"),
            "pre_recovery_crest_db": cur.get("crest_factor_db"),
            "pre_recovery_true_peak_dbtp": cur.get("approx_true_peak_dbfs"),
        })
        flat_input_threshold = _env_float("BUSY_V47_LRA_RECOVERY_INPUT_LRA_MAX", 3.0, 1.0, 6.0)
        flat_output_threshold = _env_float("BUSY_V47_LRA_RECOVERY_OUTPUT_LRA_MAX", 3.0, 1.0, 6.0)
        target_min = _lra_target_for_decision(decision or {}, mode).get("min_lra_lu", 3.0)
        too_flat = bool((in_lra_f is not None and in_lra_f <= flat_input_threshold) or (out_lra_f is not None and out_lra_f <= min(flat_output_threshold, float(target_min))))
        commercial = commercial_report if isinstance(commercial_report, dict) else (decision.get("commercial_loudness_result", {}) if isinstance(decision.get("commercial_loudness_result"), dict) else {})
        commercial_floor_not_reached = bool(
            (commercial and commercial.get("target_met") is False)
            or str((commercial or {}).get("reason") or "").startswith("commercial_floor_not_reached")
            or str((commercial or {}).get("retry_blocked_by") or "") in {"damage_score_exceeds_loudness_benefit", "no_qc_safe_louder_checkpoint"}
        )
        report["trigger"] = {
            "flat_input_threshold_lu": round(float(flat_input_threshold), 3),
            "flat_output_threshold_lu": round(float(flat_output_threshold), 3),
            "target_min_lra_lu": round(float(target_min), 3),
            "too_flat": too_flat,
            "commercial_floor_not_reached": bool(commercial_floor_not_reached),
            "commercial_reason": (commercial or {}).get("reason"),
            "commercial_retry_blocked_by": (commercial or {}).get("retry_blocked_by"),
        }
        if out_lra_f is None:
            report["reason"] = "lra_recovery_lra_unavailable_after_full_analysis"
            return final, report
        if _env_bool("BUSY_V48_LRA_CANDIDATE_GENERATOR", True):
            # v48: do not try to inflate LRA as a destructive final post-process.
            # The research direction is to preserve LRA while generating and
            # repairing candidates, then let PDE/QC choose the least-damaged one.
            feasibility = build_lra_recovery_feasibility(before or {}, cur or {}, decision if isinstance(decision, dict) else {})
            report.update({
                "schema_version": "busy_lra_recovery_stage_v8_5_3_48_0_candidate_generator",
                "active": bool(too_flat),
                "attempted": False,
                "applied": False,
                "candidate_generator_mode": True,
                "feasibility": feasibility,
                "reason": "v48_candidate_generator_mode_postprocess_bypassed",
                "policy": "LRA is handled by section-aware/loudness candidate generation and near-miss repair; final post-process expansion is bypassed to avoid LUFS loss without LRA gain.",
            })
            return final, report
        if not too_flat:
            report["reason"] = "lra_recovery_not_needed"
            return final, report

        # Amount scales up only slightly for very flat outputs.  It is capped far
        # below a normal transient enhancer so it cannot reshape the song.
        gap = max(0.0, min(float(target_min), flat_output_threshold) - out_lra_f)
        amount = _env_float("BUSY_V47_LRA_RECOVERY_AMOUNT", 0.035 + min(0.035, gap * 0.018), 0.0, 0.09)
        density_mix = _env_float("BUSY_V47_LRA_RECOVERY_DENSITY_MIX", 0.012, 0.0, 0.035)
        report["active"] = True
        report["planned"] = {
            "transient_micro_expander_amount": round(float(amount), 4),
            "upward_parallel_density_mix": round(float(density_mix), 4),
            "ceiling_dbtp": round(float(FINAL_TRUE_PEAK_CEILING_DB), 3),
        }
        if amount <= 0.0001 and density_mix <= 0.0001:
            report["reason"] = "amount_zero"
            return final, report

        cand = np.asarray(final, dtype=np.float32)
        if amount > 0.0001:
            cand = transient_micro_expander(cand, sr, amount=float(amount))
        if density_mix > 0.0001:
            # Extremely low mix: used as micro detail recovery only, not as a
            # loudness push and not allowed if QC deteriorates.
            cand = upward_parallel_density(cand, sr, threshold_db=-44.0, ratio=1.18, mix=float(density_mix))
        cand = lookahead_limiter(cand, sr, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=1.0, release_ms=180)
        cand, cand_tp_gain = _normalize_to_true_peak(cand, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=_qc_oversample_factor())
        cand_analysis = analyze_audio_fast_qc(cand, sr, true_peak_oversample=_qc_oversample_factor())
        cur_lra = float(out_lra_f)
        cand_lra = float(_analysis_lra_lu(cand_analysis) or cur_lra)
        cur_lufs = float(cur.get("integrated_lufs", -99.0) or -99.0)
        cand_lufs = float(cand_analysis.get("integrated_lufs", -99.0) or -99.0)
        cur_crest = float(cur.get("crest_factor_db", 0.0) or 0.0)
        cand_crest = float(cand_analysis.get("crest_factor_db", 0.0) or 0.0)
        cur_reasons = set(_safety_risk(before, cur, decision) + _lra_guard_summary(before, cur, decision, mode).get("warnings", []))
        cand_reasons = set(_safety_risk(before, cand_analysis, decision) + _lra_guard_summary(before, cand_analysis, decision, mode).get("warnings", []))
        new_reasons = sorted(cand_reasons - cur_reasons)
        hard_new = [r for r in new_reasons if r in {"true_peak_too_hot", "correlation_too_low", "side_low_bloom", "lra_too_wide", "lra_reduced_too_much"}]
        lra_gain = cand_lra - cur_lra
        crest_gain = cand_crest - cur_crest
        lufs_loss = cur_lufs - cand_lufs
        accepted = bool(
            not hard_new
            and lufs_loss <= _env_float("BUSY_V47_LRA_RECOVERY_MAX_LUFS_LOSS", 0.35, 0.0, 1.0)
            and (lra_gain >= _env_float("BUSY_V47_LRA_RECOVERY_MIN_LRA_GAIN", 0.04, 0.0, 0.5) or crest_gain >= 0.04)
            and len(cand_reasons) <= len(cur_reasons) + 1
        )
        report["candidate"] = {
            "lra_lu": round(float(cand_lra), 3),
            "lra_gain_lu": round(float(lra_gain), 3),
            "lufs": round(float(cand_lufs), 3),
            "lufs_loss_db": round(float(lufs_loss), 3),
            "crest_db": round(float(cand_crest), 3),
            "crest_gain_db": round(float(crest_gain), 3),
            "true_peak_dbtp": round(float(cand_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0), 3),
            "tp_normalize_gain_db": round(float(cand_tp_gain), 3),
            "new_reasons": new_reasons,
            "hard_new_reasons": hard_new,
        }
        if accepted:
            report.update({"applied": True, "reason": "candidate_improved_lra_or_crest_without_hard_qc_regression"})
            return np.nan_to_num(cand, nan=0.0, posinf=0.0, neginf=0.0), report
        report.update({"applied": False, "reason": "candidate_rejected_by_lra_recovery_guard"})
        return final, report
    except Exception as exc:
        report.update({"active": False, "applied": False, "reason": "exception", "error": str(exc)[:300]})
        return final, report





def _v57_attach_damage_aware_selection_summary(report: dict[str, Any]) -> None:
    if not isinstance(report, dict):
        return
    try:
        attempts = [a for a in (report.get("render_attempts") or []) if isinstance(a, dict)]
        rejected = []
        penalized = []
        for a in attempts:
            gate = a.get("v57_damage_aware_candidate_gate") if isinstance(a.get("v57_damage_aware_candidate_gate"), dict) else {}
            if not gate:
                acceptance = a.get("acceptance") if isinstance(a.get("acceptance"), dict) else {}
                gate = acceptance.get("v57_damage_aware_candidate_gate") if isinstance(acceptance.get("v57_damage_aware_candidate_gate"), dict) else {}
            if not isinstance(gate, dict) or not gate:
                continue
            row = {
                "name": a.get("name"),
                "candidate_label": a.get("candidate_label"),
                "push_db": a.get("push_db"),
                "candidate_lufs": gate.get("candidate_lufs"),
                "candidate_lra_lu": gate.get("candidate_lra_lu"),
                "candidate_crest_db": gate.get("candidate_crest_db"),
                "blockers": gate.get("blockers", []),
                "penalties": gate.get("penalties", []),
                "penalty_score": gate.get("penalty_score"),
            }
            if gate.get("reject"):
                rejected.append(row)
            elif gate.get("penalties"):
                penalized.append(row)
        final_grade_gate = report.get("selected_candidate_final_grade_v57_damage_gate") if isinstance(report.get("selected_candidate_final_grade_v57_damage_gate"), dict) else {}
        final_grade_row = None
        if isinstance(final_grade_gate, dict) and final_grade_gate:
            final_grade_row = {
                "name": "selected_candidate_final_grade_render",
                "candidate_label": report.get("selected"),
                "push_db": None,
                "candidate_lufs": final_grade_gate.get("candidate_lufs"),
                "candidate_lra_lu": final_grade_gate.get("candidate_lra_lu"),
                "candidate_crest_db": final_grade_gate.get("candidate_crest_db"),
                "blockers": final_grade_gate.get("blockers", []),
                "penalties": final_grade_gate.get("penalties", []),
                "penalty_score": final_grade_gate.get("penalty_score"),
                "stage": "final_grade_selected_candidate_handoff",
            }
            if final_grade_gate.get("reject"):
                rejected.append(final_grade_row)
            elif final_grade_gate.get("penalties"):
                penalized.append(final_grade_row)
        delivered_final_gate = report.get("delivered_final_v57_damage_gate") if isinstance(report.get("delivered_final_v57_damage_gate"), dict) else {}
        selected_name = str(report.get("selected") or "")
        report["v57_damage_aware_safe_selection"] = {
            "schema_version": "busy_v57_damage_aware_safe_selection_summary_v8_5_3_57_2",
            "enabled": bool(_env_bool("BUSY_V57_DAMAGE_AWARE_SAFE_SELECTION", True)),
            "active": True,
            "selected": report.get("selected"),
            "selected_score": report.get("selected_score"),
            "target_met": report.get("target_met"),
            "safe_checkpoint_count": report.get("safe_checkpoint_count"),
            "render_attempt_count": report.get("render_attempt_count"),
            "rejected_candidate_count": len(rejected),
            "penalized_candidate_count": len(penalized),
            "rejected_candidates": rejected[:8],
            "penalized_candidates": penalized[:8],
            "final_grade_gate": final_grade_gate,
            "final_grade_rejected": bool(final_grade_gate.get("reject")) if isinstance(final_grade_gate, dict) else False,
            "delivered_final_gate": delivered_final_gate,
            "delivered_final_has_damage": bool(delivered_final_gate.get("reject") or delivered_final_gate.get("blockers")) if isinstance(delivered_final_gate, dict) else False,
            "selected_candidate_was_v57_rejected": any(str(x.get("name") or "") == selected_name or str(x.get("candidate_label") or "") == selected_name for x in rejected),
            "policy": "Select the best measured safe checkpoint, but reject/penalize louder candidates that create LRA collapse or transient damage. v57.1 re-checks selected-candidate final-grade render; v57.2 also records the delivered-final gate and repairs LRA feasibility diagnostics from full output/LRA guard measurements.",
        }
    except Exception as exc:
        report["v57_damage_aware_safe_selection"] = {
            "schema_version": "busy_v57_damage_aware_safe_selection_summary_v8_5_3_57",
            "active": False,
            "reason": "exception",
            "error": str(exc)[:240],
        }



def _v57_2_float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return None


def _v57_2_lra_from_guard(lra_guard: dict[str, Any] | None, output: bool = True) -> float | None:
    if not isinstance(lra_guard, dict):
        return None
    keys = ("output_lra_lu", "current_lra_lu", "final_lra_lu", "lra_lu") if output else ("input_lra_lu", "before_lra_lu", "source_lra_lu")
    for key in keys:
        val = _v57_2_float_or_none(lra_guard.get(key))
        if val is not None:
            return val
    try:
        nested = lra_guard.get("lra") if isinstance(lra_guard.get("lra"), dict) else {}
        for key in keys:
            val = _v57_2_float_or_none(nested.get(key))
            if val is not None:
                return val
    except Exception:
        pass
    return None


def _v57_2_resolve_output_lra(final_analysis: dict[str, Any] | None, lra_guard: dict[str, Any] | None = None) -> tuple[float | None, str | None]:
    val = _analysis_lra_lu(final_analysis if isinstance(final_analysis, dict) else {})
    if val is not None:
        return float(val), "output_analysis"
    val = _v57_2_lra_from_guard(lra_guard, output=True)
    if val is not None:
        return float(val), "lra_guard.output_lra_lu"
    return None, None


def _v57_2_resolve_input_lra(before: dict[str, Any] | None, lra_guard: dict[str, Any] | None = None) -> tuple[float | None, str | None]:
    val = _analysis_lra_lu(before if isinstance(before, dict) else {})
    if val is not None:
        return float(val), "input_analysis"
    val = _v57_2_lra_from_guard(lra_guard, output=False)
    if val is not None:
        return float(val), "lra_guard.input_lra_lu"
    return None, None


def _v57_2_patch_feasibility_lra_from_delivered_final(
    report: dict[str, Any],
    before: dict[str, Any] | None,
    final_analysis: dict[str, Any] | None,
    lra_guard: dict[str, Any] | None,
) -> None:
    """v57.2: repair feasibility reporting when early base analysis lacked LRA.

    Candidate feasibility is computed before the delivered full-output LRA guard is
    available.  Some fast-QC/base summaries do not carry LRA, which previously
    produced ``lra_unknown_conservative_budget`` even though the final report had
    a valid output LRA.  This does not re-open push after the fact; it rewrites the
    diagnostic reason to the measured cause so Supabase/next-run learning sees
    "measured LRA too flat" instead of a false unknown.
    """
    if not isinstance(report, dict):
        return
    feasibility = report.get("v57_commercial_loudness_feasibility")
    if not isinstance(feasibility, dict) or not feasibility:
        return
    out_lra, out_src = _v57_2_resolve_output_lra(final_analysis, lra_guard)
    in_lra, in_src = _v57_2_resolve_input_lra(before, lra_guard)
    if out_lra is None:
        feasibility.setdefault("v57_2_lra_resolution", {"resolved": False, "reason": "no_delivered_output_lra_available"})
        return
    hard_lra_floor = _env_float("BUSY_V57_HARD_LRA_FLOOR_LU", 1.05, 0.2, 4.0)
    soft_lra_floor = _env_float("BUSY_V57_SOFT_LRA_FLOOR_LU", 1.55, 0.4, 5.0)
    min_useful = _env_float("BUSY_V57_MIN_USEFUL_SAFE_PUSH_DB", 0.18, 0.0, 0.8)
    reasons = [str(x) for x in (feasibility.get("reasons") or []) if x]
    reasons = [r for r in reasons if r != "lra_unknown_conservative_budget"]
    budgets = feasibility.get("budget_by_axis_db") if isinstance(feasibility.get("budget_by_axis_db"), dict) else {}
    budgets = dict(budgets)
    if out_lra < hard_lra_floor:
        budgets["lra_db"] = 0.0
        if "lra_too_flat_measured" not in reasons:
            reasons.append("lra_too_flat_measured")
    elif out_lra < soft_lra_floor:
        budgets["lra_db"] = min(float(budgets.get("lra_db", 0.25) or 0.25), 0.25)
        if "lra_near_flat_floor_measured" not in reasons:
            reasons.append("lra_near_flat_floor_measured")
    else:
        if "lra_resolved_from_delivered_final" not in reasons:
            reasons.append("lra_resolved_from_delivered_final")
    lra_delta = None if in_lra is None else float(out_lra) - float(in_lra)
    feasibility["schema_version"] = "busy_v57_commercial_loudness_feasibility_v8_5_3_57_2"
    feasibility["current_lra_lu"] = round(float(out_lra), 3)
    feasibility["input_lra_lu"] = round(float(in_lra), 3) if in_lra is not None else feasibility.get("input_lra_lu")
    feasibility["lra_delta_lu"] = round(float(lra_delta), 3) if lra_delta is not None else feasibility.get("lra_delta_lu")
    feasibility["budget_by_axis_db"] = {k: round(float(v), 3) for k, v in budgets.items() if _v57_2_float_or_none(v) is not None}
    budget = max(0.0, min(float(v) for v in budgets.values())) if budgets else 0.0
    useful = bool(feasibility.get("enabled") and budget >= min_useful)
    feasibility["safe_push_budget_db"] = round(float(budget), 3)
    feasibility["useful_safe_push_available"] = bool(useful)
    feasibility["reasons"] = sorted(set(reasons)) or ["safe_push_budget_available"]
    feasibility["decision"] = "allow_bounded_safe_push" if useful else "do_not_push_select_best_safe_candidate"
    feasibility["v57_2_lra_resolution"] = {
        "resolved": True,
        "output_lra_source": out_src,
        "input_lra_source": in_src,
        "output_lra_lu": round(float(out_lra), 3),
        "input_lra_lu": round(float(in_lra), 3) if in_lra is not None else None,
        "lra_delta_lu": round(float(lra_delta), 3) if lra_delta is not None else None,
        "policy": "Use delivered full-output/LRA-guard measurement to distinguish measured flatness from true unknown-LRA conservatism. This is diagnostic and does not reopen push after final selection.",
    }
    if (not useful) and any(r in feasibility["reasons"] for r in ("lra_too_flat_measured", "lra_near_flat_floor_measured")):
        if str(report.get("reason") or "").startswith("v57_no_safe_push_budget"):
            report["reason"] = "v57_measured_lra_too_flat_safe_push_budget_exhausted"
            report["retry_blocked_by"] = feasibility.get("reasons")


def _v57_2_attach_delivered_final_damage_gate(
    report: dict[str, Any],
    before: dict[str, Any] | None,
    final_analysis: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    mode: str,
    lra_guard: dict[str, Any] | None,
    warnings: list[str] | None,
) -> None:
    """Record a v57 damage gate for the actual delivered final, even with no selected candidate.

    v57.1 gated only selected-candidate final-grade handoff.  When feasibility
    blocks all hot candidates and the standard final is delivered, we still need
    to log the delivered final's measured LRA/crest condition so the report and
    Supabase do not look inactive or ambiguous.
    """
    if not isinstance(report, dict):
        return
    try:
        profile = _commercial_loudness_finish_profile(decision if isinstance(decision, dict) else {}, mode)
        delivered = _analysis_summary_for_loudness(final_analysis if isinstance(final_analysis, dict) else {})
        out_lra, out_src = _v57_2_resolve_output_lra(final_analysis, lra_guard)
        if out_lra is not None and delivered.get("lra_lu") is None:
            delivered["lra_lu"] = float(out_lra)
        base_ref = report.get("standard_final_reference_analysis") if isinstance(report.get("standard_final_reference_analysis"), dict) else {}
        if not base_ref:
            base_ref = report.get("output") if isinstance(report.get("output"), dict) else delivered
        floor = float(report.get("commercial_floor_lufs", report.get("target_lufs", -9.0)) or -9.0)
        target = float(report.get("target_lufs", -8.0) or -8.0)
        delivered_gate = _v57_damage_aware_candidate_gate(
            before if isinstance(before, dict) else {},
            delivered,
            base_ref,
            decision if isinstance(decision, dict) else {},
            profile,
            list(warnings or []),
            floor,
            target,
        )
        delivered_gate["schema_version"] = "busy_v57_delivered_final_damage_gate_v8_5_3_57_2"
        delivered_gate["stage"] = "delivered_final_after_export_lock_lra_guard"
        delivered_gate["lra_measurement_source"] = out_src
        delivered_gate["policy"] = "Diagnosis-only delivered-final gate: records whether the actual returned master is already too flat/damaged, even when no selected commercial candidate exists. It does not trigger another render or rollback."
        report["delivered_final_v57_damage_gate"] = delivered_gate
        report["delivered_final_analysis_summary"] = delivered
        _v57_2_patch_feasibility_lra_from_delivered_final(report, before if isinstance(before, dict) else {}, final_analysis if isinstance(final_analysis, dict) else {}, lra_guard if isinstance(lra_guard, dict) else {})
        _v57_attach_damage_aware_selection_summary(report)
    except Exception as exc:
        report["delivered_final_v57_damage_gate"] = {
            "schema_version": "busy_v57_delivered_final_damage_gate_v8_5_3_57_2",
            "active": False,
            "reason": "exception",
            "error": str(exc)[:240],
        }


def _attach_loudness_checkpoint_diversity_report(report: dict[str, Any]) -> None:
    """v8.5.3.47 expose floor-2/floor-1/floor checkpoint diversity audit."""
    if not isinstance(report, dict):
        return
    try:
        floor = float(report.get("commercial_floor_lufs", report.get("target_lufs", -9.0)) or -9.0)
        target = float(report.get("target_lufs", floor) or floor)
        attempts = [a for a in (report.get("render_attempts") or []) if isinstance(a, dict)]

        def _first_metric(src: dict[str, Any], *keys: str) -> Any:
            if not isinstance(src, dict):
                return None
            for key in keys:
                if src.get(key) is not None:
                    return src.get(key)
            return None

        def _nested_metric(src: dict[str, Any], parent: str, key: str) -> Any:
            obj = src.get(parent) if isinstance(src, dict) else None
            return obj.get(key) if isinstance(obj, dict) and obj.get(key) is not None else None

        checkpoints = []
        for a in attempts:
            an = a.get("analysis") if isinstance(a.get("analysis"), dict) else {}
            if not an and isinstance(a.get("output"), dict):
                an = a.get("output")
            lufs = _first_metric(an, "integrated_lufs", "lufs")
            true_peak = _first_metric(an, "approx_true_peak_dbfs", "true_peak_dbtp")
            crest = _first_metric(an, "crest_factor_db", "crest_db")
            lra = _first_metric(an, "lra_lu", "loudness_range_lu", "loudness_range")
            corr = _first_metric(an, "correlation")
            harsh = _first_metric(an, "harshness_index_db")
            if harsh is None:
                harsh = _nested_metric(an, "quality_indices", "harshness_index_db")
            air = _first_metric(an, "air_index_db")
            if air is None:
                air = _nested_metric(an, "quality_indices", "air_index_db")
            side_low = _first_metric(an, "side_low_ratio")
            if side_low is None:
                side_low = _nested_metric(an, "ms", "side_ratio_lowband")
            side_high = _first_metric(an, "side_high_ratio")
            if side_high is None:
                side_high = _nested_metric(an, "ms", "side_ratio_8k_14k")
            checkpoints.append({
                "name": a.get("name"),
                "candidate_label": a.get("candidate_label"),
                "attempt_type": a.get("attempt_type"),
                "push_db": a.get("push_db"),
                "integrated_lufs": round(float(lufs), 3) if lufs is not None else None,
                "true_peak_dbtp": round(float(true_peak), 3) if true_peak is not None else None,
                "lra_lu": round(float(lra), 3) if lra is not None else None,
                "crest_factor_db": round(float(crest), 3) if crest is not None else None,
                "correlation": round(float(corr), 4) if corr is not None else None,
                "side_low_ratio": side_low,
                "side_high_ratio": side_high,
                "harshness_index_db": harsh,
                "air_index_db": air,
                "qc_passed_safe_checkpoint": bool(a.get("qc_passed_safe_checkpoint")),
                "reject_reason": a.get("candidate_rejection_reason") or a.get("retry_blocked_by") or a.get("reason"),
                "hard_blocks": a.get("hard_blocks", []),
                "unresolved_blocker_set": a.get("unresolved_blocker_set", []),
                "score": a.get("score") or ((a.get("selection_score") or {}).get("score_0_100") if isinstance(a.get("selection_score"), dict) else None),
                "_attempt_pde": a.get("pde") if isinstance(a.get("pde"), dict) else None,
            })
        target_bins = [round(floor - 2.0, 3), round(floor - 1.0, 3), round(floor, 3), round(target, 3)]
        base_summary = report.get("input") if isinstance(report.get("input"), dict) else {}
        profile_summary = {
            "family": report.get("family"),
            "profile": report.get("family"),
            "hard_crest_min_db": report.get("hard_crest_min_db"),
            "crest_min_db": report.get("crest_min_db"),
            "mastering_intensity": report.get("mastering_intensity"),
        }
        pde_items = []
        for cp in checkpoints:
            lufs = cp.get("integrated_lufs")
            if lufs is None:
                cp["nearest_checkpoint_bin"] = None
            else:
                cp["nearest_checkpoint_bin"] = min(target_bins, key=lambda b: abs(float(lufs) - float(b)))
            if _env_bool("BUSY_V48_PDE", True) and isinstance(base_summary, dict) and base_summary:
                cand_summary = {
                    "integrated_lufs": cp.get("integrated_lufs"),
                    "true_peak_dbtp": cp.get("true_peak_dbtp"),
                    "lra_lu": cp.get("lra_lu"),
                    "crest_factor_db": cp.get("crest_factor_db"),
                    "correlation": cp.get("correlation"),
                    "quality_indices": {
                        "harshness_index_db": cp.get("harshness_index_db"),
                        "air_index_db": cp.get("air_index_db"),
                    },
                    "ms": {
                        "side_ratio_lowband": cp.get("side_low_ratio"),
                        "side_ratio_8k_14k": cp.get("side_high_ratio"),
                    },
                }
                pde = cp.get("_attempt_pde") if isinstance(cp.get("_attempt_pde"), dict) else None
                if not isinstance(pde, dict):
                    pde = build_pde_score(base_analysis=base_summary, candidate_analysis=cand_summary, profile=profile_summary, context={})
                blockers = list(cp.get("hard_blocks") or []) + list(cp.get("unresolved_blocker_set") or [])
                nm = build_near_miss_eligibility(
                    base_analysis=base_summary,
                    candidate_analysis=cand_summary,
                    blockers=blockers,
                    pde=pde,
                    min_lufs_advantage=_env_float("BUSY_V48_NEARMISS_MIN_LUFS_ADVANTAGE", 0.6, 0.1, 2.0),
                    profile=profile_summary,
                    max_crest_deficit_db=_env_float("BUSY_V48_NEARMISS_MAX_CREST_DEFICIT_DB", 0.5, 0.0, 2.0),
                )
                cp.pop("_attempt_pde", None)
                cp["pde"] = pde
                cp["nearmiss_eligibility"] = nm
                pde_items.append({"name": cp.get("name"), "pde_ratio": pde.get("ratio"), "decision_zone": pde.get("decision_zone"), "hard_vetoes": pde.get("hard_vetoes", []), "near_miss": nm.get("eligible")})
        for _cp in checkpoints:
            if isinstance(_cp, dict):
                _cp.pop("_attempt_pde", None)
        report["pde"] = {
            "schema_version": "busy_pde_commercial_finish_summary_v8_5_3_48_0_1",
            "enabled": bool(_env_bool("BUSY_V48_PDE", True)),
            "items": pde_items,
            "policy": "PDE audits every measured checkpoint and exposes near-miss eligibility before any final selection change.",
        }
        report["loudness_checkpoints"] = checkpoints[:24]
        report["loudness_checkpoint_diversity"] = {
            "schema_version": "busy_loudness_checkpoint_diversity_v8_5_3_47_1",
            "enabled": bool(_env_bool("BUSY_V47_LOUDNESS_CHECKPOINT_DIVERSITY", True)),
            "active": bool(checkpoints),
            "target_bins_lufs": target_bins,
            "attempt_count": len(attempts),
            "safe_checkpoint_count": len(report.get("safe_checkpoints") or []),
            "selected": report.get("selected"),
            "selected_score": report.get("selected_score"),
            "policy": "v47.1 compares a wider set of measured loudness checkpoints by deterministic QC; unsafe hot attempts remain render_attempts and cannot become final masters",
        }
    except Exception as exc:
        report["loudness_checkpoint_diversity"] = {"enabled": True, "active": False, "reason": "exception", "error": str(exc)[:300]}


def _risk_metrics(before: dict[str, Any], after: dict[str, Any]) -> dict[str, float]:
    bq = before.get("quality_indices", {})
    aq = after.get("quality_indices", {})
    return {
        "harshness_delta_db": float(aq.get("harshness_index_db", 0) or 0) - float(bq.get("harshness_index_db", 0) or 0),
        "air_delta_db": float(aq.get("air_index_db", 0) or 0) - float(bq.get("air_index_db", 0) or 0),
        "crest_db": float(after.get("crest_factor_db", 99) or 99),
        "correlation_avg": float(after.get("correlation", 1.0) or 1.0),
        "true_peak_dbtp": float(after.get("approx_true_peak_dbfs", -999) or -999),
        "side_ratio_lowband": float(after.get("ms", {}).get("side_ratio_lowband", 0) or 0),
        "side_ratio_highband": float(after.get("ms", {}).get("side_ratio_8k_14k", 0) or 0),
        "input_lra_lu": float(_analysis_lra_lu(before) or 0.0),
        "output_lra_lu": float(_analysis_lra_lu(after) or 0.0),
        "lra_delta_lu": float((_analysis_lra_lu(after) or 0.0) - (_analysis_lra_lu(before) or 0.0)),
    }


def _safety_risk(before: dict[str, Any], after: dict[str, Any], decision: dict[str, Any]) -> list[str]:
    metrics = _risk_metrics(before, after)
    targets = decision.get("targets", {}) if isinstance(decision, dict) else {}
    crest_floor = float(targets.get("crest_floor_db", 5.2))
    corr_floor = float(targets.get("correlation_floor", 0.08))
    harsh_max = float(targets.get("harshness_delta_max_db", 1.4))
    reasons: list[str] = []
    if metrics["true_peak_dbtp"] > FINAL_TRUE_PEAK_CEILING_DB + 0.05:
        reasons.append("true_peak_too_hot")
    if metrics["correlation_avg"] < corr_floor:
        reasons.append("stereo_correlation_too_low")
    if metrics["crest_db"] < crest_floor:
        reasons.append("crest_factor_too_low")
    if metrics["harshness_delta_db"] > harsh_max:
        reasons.append("harshness_increased")
    if _env_bool("BUSY_LRA_GUARD", True):
        lra_target = _lra_target_for_decision(decision)
        out_lra = _analysis_lra_lu(after)
        in_lra = _analysis_lra_lu(before)
        th = _lra_guard_thresholds()
        if out_lra is not None and out_lra < float(lra_target["min_lra_lu"]) - max(0.25, th["flat_margin"]):
            reasons.append("lra_too_flat")
        if in_lra is not None and out_lra is not None and (out_lra - in_lra) < min(-3.8, th["delta_floor"] - 0.3):
            reasons.append("lra_reduced_too_much")
    low_side_max = float(targets.get("low_end_side_ratio_max", 0.18))
    shimmer_max = float(targets.get("shimmer_growth_ratio_max", 1.18))
    if metrics["side_ratio_lowband"] > low_side_max:
        reasons.append("low_end_too_wide")
    if metrics["correlation_avg"] < 0.03 and metrics["side_ratio_highband"] > 0.45:
        reasons.append("mono_collapse_risk")
    if metrics["side_ratio_lowband"] > 0.16:
        reasons.append("stereo_qc_low_end_side_excess")
    if metrics["air_delta_db"] > 2.5 and metrics["side_ratio_highband"] > 0.7:
        reasons.append("side_high_shimmer_increased")
    if metrics["air_delta_db"] > 1.25 and shimmer_max <= 1.08 and metrics["side_ratio_highband"] > 0.55:
        reasons.append("v7_shimmer_guard")
    return reasons


def _collect_moves(decision: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = decision.get(key, []) if isinstance(decision, dict) else []
    if isinstance(value, list):
        return value
    return []


def _normalize_move(move: dict[str, Any]) -> dict[str, Any]:
    m = dict(move)
    if "freq" not in m:
        rng = m.get("freq_hz") or m.get("frequency_hz")
        if isinstance(rng, list):
            m["freq"] = sum(map(float, rng)) / len(rng)
        elif rng is not None:
            m["freq"] = float(rng)
    if "gain_db" not in m:
        # Character ranges may contain safe/medium/bold. Use medium by default.
        for k in ("medium_db", "safe_db", "bold_db"):
            val = m.get(k)
            if isinstance(val, list) and len(val) == 2:
                m["gain_db"] = sum(map(float, val)) / len(val)
                break
        m.setdefault("gain_db", 0.0)
    if "q" not in m:
        val = m.get("q", 1.0)
        if isinstance(val, list):
            m["q"] = sum(map(float, val)) / len(val)
    m.setdefault("target", "stereo")
    m.setdefault("type", "bell")
    return m


def _normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    d = copy.deepcopy(decision or {})
    eq = d.get("eq", {}) if isinstance(d.get("eq", {}), dict) else {}
    if "corrective_eq" not in d:
        d["corrective_eq"] = eq.get("corrective", d.get("eq_moves", []))
    if "character_eq" not in d:
        d["character_eq"] = eq.get("character", [])
    d["corrective_eq"] = [_normalize_move(m) for m in d.get("corrective_eq", [])]
    d["character_eq"] = [_normalize_move(m) for m in d.get("character_eq", [])]
    d.setdefault("dynamic_eq", [])
    d.setdefault("ms", {"mono_below_hz": 120, "low_side_reduction": 0.85, "width": 1.0})
    d.setdefault("saturation", {"apply": False})
    d.setdefault("soft_clip", {})
    d.setdefault("multiband", {})
    return d


def _boldness_scale(decision: dict[str, Any], mode: str) -> float:
    """Character EQ strength before genre-family chain multipliers.

    v6 is intentionally more characterful than v5. Rollback gates still reduce the
    result if harshness/crest/correlation fails.
    """
    confidence = float(decision.get("style_confidence", decision.get("confidence", 0.55)) or 0.55)
    level = str(decision.get("boldness_level", "auto")).lower()
    if level == "safe":
        return 0.78
    if level == "medium":
        return 1.15
    if level == "bold":
        return 1.72 if _mode_defaults(mode)["allow_bold"] else 1.10
    return float(np.interp(confidence, [0.0, 0.50, 0.66, 0.78, 0.90, 1.0], [0.55, 0.82, 1.10, 1.35, 1.62, 1.85]))


def _genre_chain_params(d: dict[str, Any], mode: str) -> dict[str, float | str]:
    """Read LLM/profile genre-chain scalars with safe clamps."""
    gc = d.get("genre_chain", {}) if isinstance(d.get("genre_chain", {}), dict) else {}
    def f(key: str, default: float, lo: float, hi: float) -> float:
        try:
            val = float(gc.get(key, default))
        except Exception:
            val = default
        return float(np.clip(val, lo, hi))
    params: dict[str, float | str] = {
        "chain_strength": f("chain_strength", 1.0, 0.55, 1.85),
        "character_eq_multiplier": f("character_eq_multiplier", 1.0, 0.55, 2.10),
        "dynamic_eq_multiplier": f("dynamic_eq_multiplier", 1.0, 0.70, 1.60),
        "multiband_multiplier": f("multiband_multiplier", 1.0, 0.45, 1.70),
        "saturation_drive_multiplier": f("saturation_drive_multiplier", 1.0, 0.25, 1.80),
        "saturation_mix_multiplier": f("saturation_mix_multiplier", 1.0, 0.25, 1.80),
        "soft_clip_drive_multiplier": f("soft_clip_drive_multiplier", 1.0, 0.0, 2.00),
        "soft_clip_mix_multiplier": f("soft_clip_mix_multiplier", 1.0, 0.0, 2.00),
        "upward_density_mix": f("upward_density_mix", 0.0, 0.0, 0.12),
        "loudness_offset_db": f("loudness_offset_db", 0.0, -0.65, 1.10),
    }
    if "Clean" in mode or "Safe" in mode or "Streaming" in mode:
        params["chain_strength"] = min(float(params["chain_strength"]), 1.05)
        params["soft_clip_drive_multiplier"] = min(float(params["soft_clip_drive_multiplier"]), 0.65)
        params["soft_clip_mix_multiplier"] = min(float(params["soft_clip_mix_multiplier"]), 0.70)
        params["loudness_offset_db"] = max(float(params["loudness_offset_db"]), 0.0)
    if "Hot" in mode:
        params["chain_strength"] = max(float(params["chain_strength"]), 1.30)
        params["loudness_offset_db"] = min(float(params["loudness_offset_db"]), -0.20)
    params["dominant_family"] = str(gc.get("dominant_family", gc.get("selected_family", "unknown")))
    return params


def _saturation_params(d: dict[str, Any], mode: str, chain: dict[str, float | str] | None = None) -> tuple[bool, float, float, float, float]:
    defaults = _mode_defaults(mode)
    chain = chain or {}
    sat = d.get("saturation", {}) or {}
    apply = bool(sat.get("apply", False))
    drive = sat.get("drive_db", 0.0)
    if isinstance(drive, list):
        drive_db = float(sum(drive) / len(drive))
        apply = apply or drive_db > 0.6
    else:
        drive_db = float(drive or 0.0)
    mix = sat.get("mix", None)
    if mix is None:
        mix_pct = sat.get("mix_pct", 0)
        if isinstance(mix_pct, list):
            mix = (sum(map(float, mix_pct)) / len(mix_pct)) / 100.0
        else:
            mix = float(mix_pct or 0) / 100.0
    mix = float(mix or 0.0)
    clip = d.get("soft_clip", {}) or {}
    clip_drive = clip.get("drive_db", None)
    if clip_drive is None:
        clip_range = sat.get("soft_clip_db", defaults["clip_drive_db"])
        if isinstance(clip_range, list):
            clip_drive = sum(map(float, clip_range)) / len(clip_range)
        else:
            clip_drive = float(clip_range or defaults["clip_drive_db"])
    clip_mix = float(clip.get("mix", defaults["clip_mix"]))
    if "Hot" in mode:
        clip_drive = max(float(clip_drive), defaults["clip_drive_db"])
        clip_mix = max(clip_mix, defaults["clip_mix"])
    elif "Clean" in mode or "Safe" in mode or "Streaming" in mode:
        clip_drive = min(float(clip_drive), 0.4)
        clip_mix = min(clip_mix, 0.10)
    drive_db *= float(chain.get("saturation_drive_multiplier", 1.0) or 1.0)
    mix *= float(chain.get("saturation_mix_multiplier", 1.0) or 1.0)
    clip_drive = float(clip_drive) * float(chain.get("soft_clip_drive_multiplier", 1.0) or 1.0)
    clip_mix = float(clip_mix) * float(chain.get("soft_clip_mix_multiplier", 1.0) or 1.0)
    drive_db = float(np.clip(drive_db, 0.0, 4.2))
    mix = float(np.clip(mix, 0.0, 0.65))
    clip_drive = float(np.clip(clip_drive, 0.0, 3.0))
    clip_mix = float(np.clip(clip_mix, 0.0, 0.55))
    return apply, float(drive_db), float(mix), float(clip_drive), float(clip_mix)




def preflight_clean_audio(y: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Input safety pass: DC offset removal, click/pop guard, and micro fade safety.

    This is deliberately conservative. It does not master the sound; it only removes
    conditions that can mislead analysis or create clicks after limiting.
    For long direct-upload WAVs, default to float32 so Stage A-1 does not double
    memory before the analysis checkpoint is written.
    """
    _dtype_name = os.environ.get("BUSY_PREFLIGHT_DTYPE", "float32").strip().lower()
    _dtype = np.float64 if _dtype_name == "float64" else np.float32
    out = np.asarray(y, dtype=_dtype)
    if out.ndim == 1:
        out = np.stack([out, out], axis=1)
    if out.shape[1] == 1:
        out = np.repeat(out, 2, axis=1)
    if out.shape[1] > 2:
        out = out[:, :2]
    report: dict[str, Any] = {"active": False, "actions": []}
    if out.size == 0:
        return out, report

    dc = np.mean(out, axis=0)
    max_dc = float(np.max(np.abs(dc)))
    report["dc_offset"] = [round(float(v), 8) for v in dc]
    if _env_bool("BUSY_DC_OFFSET_GUARD", True) and max_dc > 1e-5:
        out -= dc.reshape(1, -1)
        report["active"] = True
        report["actions"].append("dc_offset_removed")

    # Very rare pop/spike guard: only soft-clamps pathological isolated peaks.
    if _env_bool("BUSY_CLICK_POP_GUARD", True):
        peak = float(np.max(np.abs(out)))
        p9999 = float(np.percentile(np.abs(out), 99.99)) if out.size else 0.0
        if peak > 1.02 or (peak > 0.985 and p9999 > 1e-8 and peak / max(p9999, 1e-8) > 1.8):
            drive = max(1.0, peak / max(0.98, p9999))
            soft = np.tanh(out * drive) / np.tanh(drive)
            out = out * 0.72 + soft * 0.28
            report["active"] = True
            report["actions"].append("isolated_peak_soft_guard")
            report["pre_peak"] = round(peak, 6)
            report["p9999_abs"] = round(p9999, 6)

    # Click-safe micro fades at boundaries. This is not a musical fade.
    if _env_bool("BUSY_AUTO_FADE_SAFETY", True):
        fade_ms = _env_float("BUSY_AUTO_FADE_MS", 8.0, 2.0, 30.0)
        n = min(int(sr * fade_ms / 1000.0), max(0, out.shape[0] // 20))
        if n > 8:
            edge_peak = max(float(np.max(np.abs(out[:n]))), float(np.max(np.abs(out[-n:]))))
            boundary = max(float(np.max(np.abs(out[0]))), float(np.max(np.abs(out[-1]))))
            # Apply when the file starts/ends non-zero or always if configured.
            always = _env_bool("BUSY_AUTO_FADE_ALWAYS", False)
            if always or boundary > 0.004 or edge_peak > 0.75:
                fade_in = np.linspace(0.0, 1.0, n, dtype=out.dtype)[:, None]
                fade_out = np.linspace(1.0, 0.0, n, dtype=out.dtype)[:, None]
                out[:n] *= fade_in
                out[-n:] *= fade_out
                report["active"] = True
                report["actions"].append(f"micro_fade_{fade_ms:.1f}ms")
                report["edge_peak"] = round(edge_peak, 6)
                report["boundary_peak"] = round(boundary, 6)

    # v8.5.3.6: Suno-specific temporal spectral drift stabilizer.
    # Analysis uses short frames, but any correction is a continuous inverse
    # automation curve, never a block/section step EQ. It is deliberately
    # conservative and only acts on high-confidence abnormal gradual drift.
    if _env_bool("BUSY_TEMPORAL_DRIFT_STABILIZER", True):
        try:
            out, drift_report = stabilize_temporal_spectral_drift(out, sr)
            report["temporal_spectral_drift"] = drift_report
            if isinstance(drift_report, dict) and drift_report.get("active"):
                report["active"] = True
                report.setdefault("actions", []).append("temporal_spectral_drift_stabilized")
        except Exception as e:
            report["temporal_spectral_drift"] = {
                "schema_version": "busy_temporal_spectral_drift_v1",
                "enabled": True,
                "active": False,
                "reason": "error",
                "actions": [],
                "selected_corrections_count": 0,
                "error_type": type(e).__name__,
                "error": str(e)[:300],
            }

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return out, report


def _band_sos(sr: int, lo: float | None = None, hi: float | None = None, order: int = 4):
    nyq = sr * 0.5
    if lo is not None:
        lo = max(10.0, float(lo))
    if hi is not None:
        hi = min(float(hi), nyq * 0.98)
    if lo is None and hi is None:
        return None
    if lo is None:
        return butter(order, hi / nyq, btype="lowpass", output="sos")
    if hi is None:
        return butter(order, lo / nyq, btype="highpass", output="sos")
    if hi <= lo + 10:
        hi = lo + 10
    return butter(order, [lo / nyq, hi / nyq], btype="bandpass", output="sos")


def _filter_band(x: np.ndarray, sr: int, lo: float | None, hi: float | None) -> np.ndarray:
    sos = _band_sos(sr, lo=lo, hi=hi)
    if sos is None:
        return x.copy()
    return sosfiltfilt(sos, x)


def _profile_text(decision: dict[str, Any]) -> str:
    rv7 = decision.get("reference_v7", {}) if isinstance(decision.get("reference_v7", {}), dict) else {}
    return " ".join(str(x or "") for x in [
        decision.get("detected_style"), decision.get("genre_id"), decision.get("selected_profile"),
        rv7.get("selected_profile"), rv7.get("family"), decision.get("mastering_intent"),
    ]).lower()


def _runtime_context(decision: dict[str, Any]) -> dict[str, Any]:
    ctx = decision.get("runtime_safety_context", {}) if isinstance(decision.get("runtime_safety_context", {}), dict) else {}
    return ctx


def _loudness_push_governor(before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Safety-first governor that softens loudness/clip/saturation push when source is risky."""
    d = copy.deepcopy(decision or {})
    if not _env_bool("BUSY_LOUDNESS_PUSH_GOVERNOR", True):
        return d, {"active": False, "reason": "disabled"}
    q = before.get("quality_indices", {}) if isinstance(before, dict) else {}
    ms = before.get("ms", {}) if isinstance(before, dict) else {}
    artifact = before.get("suno_artifact_analysis", {}) if isinstance(before.get("suno_artifact_analysis", {}), dict) else {}
    lufs = float(before.get("integrated_lufs", -14.0) or -14.0)
    crest = float(before.get("crest_factor_db", 9.0) or 9.0)
    true_peak = float(before.get("approx_true_peak_dbfs", -2.0) or -2.0)
    side_low = float(ms.get("side_ratio_lowband", 0.0) or 0.0)
    side_high = float(ms.get("side_ratio_8k_14k", 0.0) or 0.0)
    harsh = float(q.get("harshness_index_db", 0.0) or 0.0)
    air = float(q.get("air_index_db", 0.0) or 0.0)
    artifact_risk = float(artifact.get("overall_risk", 0.0) or 0.0)
    scores = artifact.get("risk_scores", {}) if isinstance(artifact.get("risk_scores", {}), dict) else {}
    residue_ctx = residue_indirect_risk_summary(before.get("frequency_band_centrifuge") if isinstance(before, dict) else None)
    residue_risk = float(residue_ctx.get("risk_score", 0.0) or 0.0) if isinstance(residue_ctx, dict) else 0.0
    reasons: list[str] = []
    risk = 0.0
    def add(name: str, value: float) -> None:
        nonlocal risk
        value = float(np.clip(value, 0.0, 1.0))
        if value > 0.05:
            reasons.append(name)
            risk = max(risk, value)
    add("already_limited", (8.0 - crest) / 3.0 + max(0.0, -8.0 - lufs) / 8.0)
    add("true_peak_hot_input", (true_peak + 0.8) / 1.4)
    add("side_low_excess", (side_low - 0.12) / 0.18)
    add("side_high_shimmer", (side_high - 0.50) / 0.45)
    add("harsh_or_brittle", max(harsh - 1.3, air - 1.4) / 3.0)
    add("suno_artifact_risk", artifact_risk)
    add("brittle_8_12k", float(scores.get("brittle_8_12k_top", 0) or 0))
    if residue_ctx.get("active"):
        add("frequency_band_centrifuge_high_band_residue", residue_risk)

    relax_db = 0.0
    if risk >= 0.80:
        relax_db = -1.05
    elif risk >= 0.62:
        relax_db = -0.70
    elif risk >= 0.40:
        relax_db = -0.38
    elif risk >= 0.25:
        relax_db = -0.18
    # Do not over-relax a deliberately hot mode, but still add guardrails.
    if "Hot" in mode and relax_db < -0.70:
        relax_db = -0.70
    gc = d.setdefault("genre_chain", {})
    try:
        gc["loudness_offset_db"] = float(gc.get("loudness_offset_db", 0.0) or 0.0) + relax_db
        scale = float(np.interp(risk, [0.0, 0.40, 0.75, 1.0], [1.0, 0.92, 0.78, 0.68]))
        for key in ["soft_clip_drive_multiplier", "soft_clip_mix_multiplier", "saturation_drive_multiplier", "saturation_mix_multiplier"]:
            gc[key] = float(gc.get(key, 1.0) or 1.0) * scale
        gc["multiband_multiplier"] = float(gc.get("multiband_multiplier", 1.0) or 1.0) * float(np.interp(risk, [0, 1], [1.0, 0.82]))
        # v8.5.3.10: residue centrifuge is indirect-only.  When it reports
        # high-band/side residue risk, later stages should avoid making that
        # residue louder through saturation, clipping or high-band widening.
        if residue_ctx.get("active"):
            hf_scalar = float(residue_ctx.get("limiter_hf_push_scalar", 1.0) or 1.0)
            widener_scalar = float(residue_ctx.get("reduce_widener_scalar", 1.0) or 1.0)
            gc["soft_clip_drive_multiplier"] = float(gc.get("soft_clip_drive_multiplier", 1.0) or 1.0) * hf_scalar
            gc["soft_clip_mix_multiplier"] = float(gc.get("soft_clip_mix_multiplier", 1.0) or 1.0) * hf_scalar
            gc["saturation_drive_multiplier"] = float(gc.get("saturation_drive_multiplier", 1.0) or 1.0) * float(residue_ctx.get("reduce_exciter_scalar", 1.0) or 1.0)
            gc["saturation_mix_multiplier"] = float(gc.get("saturation_mix_multiplier", 1.0) or 1.0) * float(residue_ctx.get("reduce_exciter_scalar", 1.0) or 1.0)
            ms_cfg = d.setdefault("ms", {})
            try:
                ms_cfg["width"] = float(ms_cfg.get("width", 1.0) or 1.0) * float(np.clip(widener_scalar, 0.72, 1.0))
            except Exception:
                pass
    except Exception:
        pass
    ctx = {
        "active": bool(risk >= 0.25),
        "risk_score": round(float(risk), 3),
        "relax_loudness_db": round(float(relax_db), 3),
        "reasons": sorted(set(reasons)),
        "input_lufs": round(lufs, 3),
        "input_crest": round(crest, 3),
        "input_true_peak": round(true_peak, 3),
        "side_low": round(side_low, 4),
        "side_high": round(side_high, 4),
        "frequency_band_centrifuge": residue_ctx if isinstance(residue_ctx, dict) and residue_ctx.get("active") else {"active": False, "reason": (residue_ctx or {}).get("reason") if isinstance(residue_ctx, dict) else "not_available"},
    }
    d["runtime_safety_context"] = ctx
    return d, ctx


def _intelligent_multiband_imager(y: np.ndarray, sr: int, decision: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Band-aware imager: widen only safe bands and protect mono/center-critical ranges."""
    if not _env_bool("BUSY_INTELLIGENT_IMAGER", True):
        return y, {"active": False, "reason": "disabled"}
    txt = _profile_text(decision)
    ms_cfg = decision.get("ms", {}) if isinstance(decision.get("ms", {}), dict) else {}
    ctx = _runtime_context(decision)
    risk = float(ctx.get("risk_score", 0.0) or 0.0)
    residue_ctx = ctx.get("frequency_band_centrifuge", {}) if isinstance(ctx.get("frequency_band_centrifuge", {}), dict) else {}
    residue_side_risk = float(residue_ctx.get("side_high_residue_risk", 0.0) or 0.0) if residue_ctx.get("active") else 0.0
    repair = decision.get("suno_repair", {}) if isinstance(decision.get("suno_repair", {}), dict) else {}
    repair_active = bool(repair.get("active", False))
    mono_below = float(ms_cfg.get("mono_below_hz", 110.0) or 110.0)
    side_reduce = float(ms_cfg.get("side_reduce_below_hz", ms_cfg.get("mono_below_hz", 140.0)) or 140.0)
    base_width = float(ms_cfg.get("width", 1.0) or 1.0)
    if "hyperpop" in txt:
        high_gain, air_gain = 1.10, 1.06
    elif any(k in txt for k in ["edm", "festival", "house", "trance"]):
        high_gain, air_gain = 1.08, 1.04
    elif any(k in txt for k in ["drift", "phonk", "hard_techno", "schranz", "industrial"]):
        high_gain, air_gain = 1.04, 0.99
    elif any(k in txt for k in ["ballad", "spoken", "narration"]):
        high_gain, air_gain = 1.00, 0.98
    elif any(k in txt for k in ["live", "cinematic", "orchestral"]):
        high_gain, air_gain = 1.03, 1.01
    else:
        high_gain, air_gain = 1.04, 1.02
    # Risk-aware guard: side shimmer or brittle top means no high widening.
    if risk >= 0.55 or repair_active:
        high_gain = min(high_gain, 1.00)
        air_gain = min(air_gain, 0.96)
    if residue_side_risk >= 0.42:
        # Residue centrifuge is indirect-only: do not remove audio here, simply
        # stop widening the high side field that would expose side hash.
        high_gain = min(high_gain, float(np.interp(residue_side_risk, [0.42, 0.85, 1.0], [0.98, 0.90, 0.86])))
        air_gain = min(air_gain, float(np.interp(residue_side_risk, [0.42, 0.85, 1.0], [0.94, 0.84, 0.78])))
    lowmid_gain = 0.93 if risk >= 0.35 or any(k in txt for k in ["trap", "phonk", "techno", "edm", "bass"]) else 0.98
    vocal_band_gain = 0.98 if any(k in txt for k in ["kpop", "pop", "vocal", "rnb", "ballad"]) else 1.00
    if "spoken" in txt:
        high_gain, air_gain, lowmid_gain, vocal_band_gain = 0.96, 0.92, 0.90, 0.95
    high_gain *= float(np.clip(base_width, 0.90, 1.12))
    air_gain *= float(np.clip(base_width, 0.90, 1.08))

    mid, side = lr_to_ms(y)
    side_orig = side
    side_new = side.copy()
    # Hard low-side cleanup below side_reduce, in addition to existing ms_low_mono.
    try:
        low = _filter_band(side_orig, sr, None, side_reduce)
        lowmid = _filter_band(side_orig, sr, side_reduce, 350)
        vocal_band = _filter_band(side_orig, sr, 1000, 4000)
        high = _filter_band(side_orig, sr, 4500, 9000)
        air = _filter_band(side_orig, sr, 9000, 14500)
        side_new = side_new - low * 0.65 - lowmid * (1.0 - lowmid_gain) - vocal_band * (1.0 - vocal_band_gain)
        side_new = side_new + high * (high_gain - 1.0) + air * (air_gain - 1.0)
    except Exception:
        return y, {"active": False, "reason": "filter_failed"}
    out = ms_to_lr(mid, side_new)
    peak_in = float(np.max(np.abs(y))) if y.size else 0.0
    peak_out = float(np.max(np.abs(out))) if out.size else 0.0
    if peak_out > peak_in > 0:
        out = out * min(1.0, peak_in / peak_out * 1.003)
    report = {
        "active": True,
        "mono_below_hz": round(mono_below, 2),
        "side_reduce_below_hz": round(side_reduce, 2),
        "lowmid_side_gain": round(lowmid_gain, 3),
        "vocal_band_side_gain": round(vocal_band_gain, 3),
        "high_side_gain": round(high_gain, 3),
        "air_side_gain": round(air_gain, 3),
        "risk_score": round(risk, 3),
        "frequency_band_centrifuge_side_high_risk": round(float(residue_side_risk), 4),
    }
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report


def _transient_punch_guard(y: np.ndarray, sr: int, decision: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if not _env_bool("BUSY_TRANSIENT_PUNCH_GUARD", True):
        return y, {"active": False, "reason": "disabled"}
    txt = _profile_text(decision)
    ctx = _runtime_context(decision)
    if not any(k in txt for k in ["edm", "house", "trap", "phonk", "techno", "schranz", "bass", "rock"]):
        return y, {"active": False, "reason": "profile_not_transient_priority"}
    risk = float(ctx.get("risk_score", 0.0) or 0.0)
    amount = 0.035
    if any(k in txt for k in ["hard_techno", "schranz", "drift", "phonk", "edm", "trap"]):
        amount = 0.052
    if risk > 0.60:
        amount *= 0.65
    amount = float(np.clip(amount, 0.0, 0.075))
    if amount < 0.01:
        return y, {"active": False, "reason": "amount_too_low"}
    out = transient_micro_expander(y, sr, amount=amount)
    return out, {"active": True, "amount": round(amount, 4), "reason": "club_or_bass_profile"}


def _vocal_lead_protection(y: np.ndarray, sr: int, decision: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if not _env_bool("BUSY_VOCAL_LEAD_PROTECTION", True):
        return y, {"active": False, "reason": "disabled"}
    txt = _profile_text(decision)
    if not any(k in txt for k in ["kpop", "pop", "vocal", "rnb", "ballad", "commercial", "spoken", "worship"]):
        return y, {"active": False, "reason": "profile_not_vocal_forward"}
    ctx = _runtime_context(decision)
    risk = float(ctx.get("risk_score", 0.0) or 0.0)
    boost_db = 0.32 if risk < 0.60 else 0.18
    if "spoken" in txt:
        boost_db = 0.25
    moves = [
        {"type": "bell", "freq": 2300, "gain_db": boost_db, "q": 0.9, "target": "mid"},
    ]
    if risk >= 0.40 or any(k in txt for k in ["hyperpop", "kpop", "commercial"]):
        # Side air restraint keeps vocals from being pushed backward by a wide bright side image.
        moves.append({"type": "high_shelf", "freq": 7800, "gain_db": -0.25 if risk < 0.65 else -0.45, "q": 0.8, "target": "side"})
    out = apply_eq_moves(y, sr, moves, scale=1.0)
    return out, {"active": True, "mid_presence_restore_db": round(boost_db, 3), "moves": moves, "risk_score": round(risk, 3)}


def _stereo_qc_summary(analysis: dict[str, Any]) -> dict[str, Any]:
    ms = analysis.get("ms", {}) if isinstance(analysis, dict) else {}
    corr = float(analysis.get("correlation", ms.get("correlation_avg", 1.0)) or 1.0)
    low_side = float(ms.get("side_ratio_lowband", 0.0) or 0.0)
    high_side = float(ms.get("side_ratio_8k_14k", 0.0) or 0.0)
    warnings: list[str] = []
    if corr < 0.03:
        warnings.append("mono_collapse_risk")
    if low_side > 0.16:
        warnings.append("low_end_side_excess")
    if high_side > 0.68 and corr < 0.18:
        warnings.append("phasey_high_side")
    if corr > 0.97:
        warnings.append("overly_narrow_or_dual_mono")
    return {"correlation": round(corr, 4), "low_side_ratio": round(low_side, 5), "high_side_ratio": round(high_side, 5), "warnings": warnings}


def _ab_delta_qc(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    bq = before.get("quality_indices", {}) if isinstance(before, dict) else {}
    aq = after.get("quality_indices", {}) if isinstance(after, dict) else {}
    bm = before.get("ms", {}) if isinstance(before, dict) else {}
    am = after.get("ms", {}) if isinstance(after, dict) else {}
    bcrest = float(before.get("crest_factor_db", 99) or 99)
    acrest = float(after.get("crest_factor_db", 99) or 99)
    deltas = {
        "lufs_delta": round(float(after.get("integrated_lufs", 0) or 0) - float(before.get("integrated_lufs", 0) or 0), 3),
        "crest_delta": round(acrest - bcrest, 3),
        "harshness_delta_db": round(float(aq.get("harshness_index_db", 0) or 0) - float(bq.get("harshness_index_db", 0) or 0), 3),
        "air_delta_db": round(float(aq.get("air_index_db", 0) or 0) - float(bq.get("air_index_db", 0) or 0), 3),
        "low_side_delta": round(float(am.get("side_ratio_lowband", 0) or 0) - float(bm.get("side_ratio_lowband", 0) or 0), 5),
        "high_side_delta": round(float(am.get("side_ratio_8k_14k", 0) or 0) - float(bm.get("side_ratio_8k_14k", 0) or 0), 5),
        "lra_delta_lu": round(float((_analysis_lra_lu(after) or 0.0) - (_analysis_lra_lu(before) or 0.0)), 3),
    }
    warnings: list[str] = []
    if deltas["crest_delta"] < -4.2:
        warnings.append("transient_loss_delta")
    if deltas["harshness_delta_db"] > 1.8:
        warnings.append("harshness_delta_excess")
    if deltas["air_delta_db"] > 2.4 and deltas["high_side_delta"] > 0.08:
        warnings.append("side_air_delta_excess")
    if deltas["low_side_delta"] > 0.05:
        warnings.append("low_side_delta_excess")
    if _env_bool("BUSY_LRA_GUARD", True) and deltas.get("lra_delta_lu", 0.0) < min(-3.8, _lra_guard_thresholds()["delta_floor"] - 0.3):
        warnings.append("lra_reduced_too_much")
    return {"deltas": deltas, "warnings": warnings}




def _commercial_loudness_prep_plan(before: dict[str, Any], decision: dict[str, Any], mode: str) -> dict[str, Any]:
    """Build a safe pre-limiter conditioning plan for stronger commercial loudness.

    This is not a fixed +6 dB push. It prepares the signal so the limiter can work
    harder with less pumping, harshness, low-end overs, and mono/stereo failures.
    """
    if not _env_bool("BUSY_COMMERCIAL_LOUDNESS_PREP", True):
        return {"active": False, "reason": "disabled"}
    mode_l = str(mode or "").lower()
    hot_exp = _experimental_hot_commercial_enabled(mode)
    if ("streaming" in mode_l or "safe" in mode_l or "clean" in mode_l) and not _env_bool("BUSY_COMMERCIAL_LOUDNESS_PREP_IN_SAFE_MODES", False):
        return {"active": False, "reason": "safe_or_clean_mode"}

    q = before.get("quality_indices", {}) if isinstance(before, dict) else {}
    ms = before.get("ms", {}) if isinstance(before, dict) else {}
    artifact = before.get("suno_artifact_analysis", {}) if isinstance(before.get("suno_artifact_analysis", {}), dict) else {}
    crest = float(before.get("crest_factor_db", 12.0) or 12.0)
    true_peak = float(before.get("approx_true_peak_dbfs", -2.0) or -2.0)
    side_low = float(ms.get("side_ratio_lowband", 0.0) or 0.0)
    side_high = float(ms.get("side_ratio_8k_14k", 0.0) or 0.0)
    harsh = float(q.get("harshness_index_db", -6.0) or -6.0)
    air = float(q.get("air_index_db", -12.0) or -12.0)
    mud = float(q.get("mud_index_db", 0.0) or 0.0)
    bass = float(q.get("bass_index_db", 0.0) or 0.0)
    artifact_risk = float(artifact.get("overall_risk", 0.0) or 0.0)
    active_repairs = set(artifact.get("active_repairs", []) or [])
    residue_ctx = residue_indirect_risk_summary(before.get("frequency_band_centrifuge") if isinstance(before, dict) else None)
    residue_risk = float(residue_ctx.get("risk_score", 0.0) or 0.0) if isinstance(residue_ctx, dict) else 0.0
    lra_target = _lra_target_for_decision(decision, mode)
    input_lra = _analysis_lra_lu(before)

    score = 100.0
    reasons: list[str] = []
    if artifact_risk > 0.45:
        score -= min(32.0, artifact_risk * 34.0)
        reasons.append("artifact_risk")
    if residue_ctx.get("active") and residue_risk > 0.40:
        # v8.5.3.14 adaptive hot mode treats residue risk as a blocker to
        # clean lightly before pushing, not as an automatic quiet-master reason.
        if hot_exp:
            score -= min(7.0, residue_risk * 8.0)
            reasons.append("frequency_band_centrifuge_residue_risk_adaptive_cleanup")
        else:
            score -= min(18.0, residue_risk * 20.0)
            reasons.append("frequency_band_centrifuge_residue_risk")
    if side_low > 0.10:
        score -= min(18.0, (side_low - 0.10) * 95.0)
        reasons.append("side_low")
    if side_high > 0.42:
        score -= min(12.0, (side_high - 0.42) * 45.0)
        reasons.append("side_high")
    if true_peak > -1.25:
        score -= min(12.0, (true_peak + 1.25) * 9.0)
        reasons.append("limited_headroom")
    if crest < 10.0:
        score -= min(16.0, (10.0 - crest) * 4.0)
        reasons.append("low_crest")
    if harsh > -3.5:
        score -= min(14.0, (harsh + 3.5) * 3.0)
        reasons.append("harsh")
    if mud > 7.0:
        score -= min(10.0, (mud - 7.0) * 1.2)
        reasons.append("mud")
    if _env_bool("BUSY_LRA_GUARD", True) and input_lra is not None:
        if input_lra < float(lra_target["min_lra_lu"]) - 0.5:
            score -= 5.0 if hot_exp else 14.0
            reasons.append("lra_already_flat_adaptive_hot_allowed" if hot_exp else "lra_already_flat")
        elif input_lra > float(lra_target["max_lra_lu"]) + 1.0:
            score -= 4.0
            reasons.append("lra_wide")

    score = float(np.clip(score, 0.0, 100.0))
    if score >= 72:
        readiness = "high"
    elif score >= 52:
        readiness = "medium"
    else:
        readiness = "low"

    max_extra = _env_float("BUSY_MAX_EXTRA_PUSH_DB", 1.15 if hot_exp else 0.65, 0.0, 1.8 if hot_exp else 1.2)
    if readiness == "high":
        extra_push = min(max_extra, 0.55)
    elif readiness == "medium":
        extra_push = min(max_extra, 0.28)
    else:
        extra_push = min(max_extra, 0.12) if artifact_risk < 0.45 and crest > 12.0 else 0.0
    if artifact_risk >= 0.75 or "brittle_8_12k_top" in active_repairs:
        extra_push = min(extra_push, 0.45 if hot_exp else 0.18)
    if residue_ctx.get("active"):
        hf_scalar = float(residue_ctx.get("limiter_hf_push_scalar", 1.0) or 1.0)
        # Normal mode caps high-band-exposing push. Experimental mode leaves
        # more headroom because direct light residue control will run before retry.
        if hot_exp:
            extra_push = min(extra_push, max(0.0, extra_push * float(np.clip(hf_scalar, 0.82, 1.0))))
            if residue_risk >= 0.72:
                extra_push = min(extra_push, 0.65)
        else:
            extra_push = min(extra_push, max(0.0, extra_push * float(np.clip(hf_scalar, 0.55, 1.0))))
            if residue_risk >= 0.72:
                extra_push = min(extra_push, 0.12)
            elif residue_risk >= 0.55:
                extra_push = min(extra_push, 0.22)
    if _env_bool("BUSY_LRA_GUARD", True) and input_lra is not None and input_lra < float(lra_target["min_lra_lu"]) - 0.5 and not hot_exp:
        # Already flat: keep commercial prep cleanup, but do not add more level pressure.
        extra_push = 0.0
    if _env_bool("BUSY_SAFE_EXTRA_PUSH_IF_CLEAN", True) is False:
        extra_push = 0.0

    intensity = _env_float("BUSY_LOUDNESS_PREP_INTENSITY", 1.0, 0.35, 1.55)
    # Higher risk means more cleanup, but not necessarily more loudness.
    cleanup_strength = float(np.clip(0.70 + artifact_risk * 0.35 + max(0.0, side_low - 0.08) * 1.1, 0.55, 1.25)) * intensity
    peak_tame = float(np.clip(0.45 + max(0.0, true_peak + 3.0) * 0.12 + max(0.0, 12.0 - crest) * 0.06, 0.35, 1.10)) * intensity

    static_moves: list[dict[str, Any]] = []
    dynamic_moves: list[dict[str, Any]] = []

    # Low-end readiness: make limiter see fewer huge sub/808 events without thinning the song.
    dynamic_moves.append({"type": "bell", "freq": 58, "gain_db": -0.55 * peak_tame, "q": 0.85, "target": "stereo"})
    dynamic_moves.append({"type": "bell", "freq": 105, "gain_db": -0.42 * peak_tame, "q": 0.95, "target": "stereo"})
    if side_low > 0.10 or "low_end_side_excess" in active_repairs:
        static_moves.append({"type": "bell", "freq": 155, "gain_db": -0.45 * cleanup_strength, "q": 0.85, "target": "side"})
        dynamic_moves.append({"type": "bell", "freq": 95, "gain_db": -0.70 * cleanup_strength, "q": 0.90, "target": "side"})

    # Low-mid density cleanup: helps LUFS without boxy limiter pumping.
    if mud > 4.5 or bass > 14.0:
        lra_flat_multiplier = 0.72 if (_env_bool("BUSY_LRA_GUARD", True) and input_lra is not None and input_lra < float(lra_target["min_lra_lu"])) else 1.0
        static_moves.append({"type": "bell", "freq": 285, "gain_db": -0.38 * cleanup_strength * lra_flat_multiplier, "q": 0.90, "target": "stereo"})
        static_moves.append({"type": "bell", "freq": 430, "gain_db": -0.24 * cleanup_strength * lra_flat_multiplier, "q": 1.00, "target": "mid"})

    # Top readiness: prevent hard limiting from pulling Suno brittleness forward.
    if artifact_risk > 0.50 or harsh > -5.0:
        static_moves.append({"type": "bell", "freq": 3900, "gain_db": -0.22 * cleanup_strength, "q": 1.10, "target": "stereo"})
    if artifact_risk > 0.55 or "brittle_8_12k_top" in active_repairs or air > -11.0:
        dynamic_moves.append({"type": "bell", "freq": 8600, "gain_db": -0.55 * cleanup_strength, "q": 1.35, "target": "stereo"})
        static_moves.append({"type": "bell", "freq": 11200, "gain_db": -0.20 * cleanup_strength, "q": 1.10, "target": "side"})

    # v8.5.3.10: Frequency-Band Centrifuge can ask for indirect high-band
    # caution.  This is not a residue-removal pass; it only prevents later
    # commercial push from making side/top hash more audible.
    if residue_ctx.get("active") and residue_risk >= 0.45 and _env_bool("BUSY_RESIDUE_DIRECT_GUARD", False):
        # Experimental opt-in only.  Default v1 behavior is indirect-only, so
        # this block is disabled unless explicitly enabled by environment.
        side_risk = float(residue_ctx.get("side_high_residue_risk", 0.0) or 0.0)
        top_scale = float(np.clip(0.45 + residue_risk * 0.55, 0.45, 0.95))
        if side_risk >= 0.42 and not any(m.get("target") == "side" and float(m.get("freq", 0) or 0) >= 8000 for m in dynamic_moves):
            dynamic_moves.append({"type": "bell", "freq": 10400, "gain_db": -0.28 * top_scale, "q": 1.05, "target": "side", "source": "frequency_band_centrifuge_direct_guard_opt_in"})
        static_moves.append({"type": "high_shelf", "freq": 11800, "gain_db": -0.08 * top_scale, "q": 0.70, "target": "side", "source": "frequency_band_centrifuge_direct_guard_opt_in"})

    micro_clip_drive = float(np.clip(0.25 + peak_tame * 0.42, 0.20, 0.90))
    micro_clip_mix = float(np.clip(0.055 + peak_tame * 0.055, 0.045, 0.16))
    if residue_ctx.get("active"):
        hf_scalar = float(residue_ctx.get("limiter_hf_push_scalar", 1.0) or 1.0)
        micro_clip_drive *= float(np.clip(hf_scalar, 0.70, 1.0))
        micro_clip_mix *= float(np.clip(hf_scalar, 0.72, 1.0))

    return {
        "active": True,
        "schema_version": "busy_commercial_loudness_prep_v8_5_3_14_adaptive_hot" if hot_exp else "busy_commercial_loudness_prep_v8_5_3_12",
        "experimental_hot_commercial": bool(hot_exp),
        "adaptive_hot_commercial": bool(hot_exp),
        "readiness_score": round(score, 1),
        "readiness": readiness,
        "reasons": reasons,
        "extra_push_db": round(float(extra_push), 3),
        "cleanup_strength": round(float(cleanup_strength), 3),
        "peak_tame": round(float(peak_tame), 3),
        "static_moves": static_moves,
        "dynamic_moves": dynamic_moves,
        "micro_clip_drive_db": round(micro_clip_drive, 3),
        "micro_clip_mix": round(micro_clip_mix, 4),
        "lra_guard": {
            "active": _env_bool("BUSY_LRA_GUARD", True),
            "target": lra_target,
            "input_lra_lu": round(input_lra, 3) if input_lra is not None else None,
            "extra_push_lra_limited": bool(_env_bool("BUSY_LRA_GUARD", True) and input_lra is not None and input_lra < float(lra_target["min_lra_lu"]) - 0.5),
        },
        "frequency_band_centrifuge_influence": residue_ctx if isinstance(residue_ctx, dict) and residue_ctx.get("active") else {"active": False, "reason": (residue_ctx or {}).get("reason") if isinstance(residue_ctx, dict) else "not_available"},
        "inputs": {
            "crest_factor_db": round(crest, 3),
            "true_peak_dbfs": round(true_peak, 3),
            "side_low": round(side_low, 5),
            "side_high": round(side_high, 5),
            "harshness_index_db": round(harsh, 3),
            "mud_index_db": round(mud, 3),
            "bass_index_db": round(bass, 3),
            "artifact_risk": round(artifact_risk, 3),
            "input_lra_lu": round(input_lra, 3) if input_lra is not None else None,
        },
    }


def _attach_commercial_loudness_prep(before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    hp = _bamix_handoff_policy_from_decision(decision)
    handoff_mode = _bamix_v6319_handoff_mode(decision)
    d = copy.deepcopy(decision or {})
    if hp.get("active") and handoff_mode in {"transparent_polish", "fragile_protection"} and _env_bool("BUSY_BAMIX_SKIP_PRELIMITER_COMMERCIAL_LOUDNESS_PREP", True):
        plan = {
            "schema_version": "busy_commercial_loudness_prep_v8_5_3_63_1_9_bamix_bypass",
            "active": False,
            "bypassed": True,
            "reason": "bamix_transparent_or_fragile_handoff_skips_raw_stereo_loudness_prep",
            "handoff_mode": handoff_mode,
            "extra_push_db": 0.0,
            "policy": "BAMix transparent/fragile handoff skips raw-stereo pre-limiter commercial prep; density-completion modes may still use bounded prep under existing QC gates.",
        }
        d["commercial_loudness_prep"] = plan
        return d, plan
    plan = _commercial_loudness_prep_plan(before, decision, mode)
    if hp.get("active") and handoff_mode in {"commercial_density_completion", "guarded_completion"}:
        plan = dict(plan) if isinstance(plan, dict) else {"active": False, "reason": "invalid_plan"}
        cap_default = 0.62 if handoff_mode == "commercial_density_completion" else 0.38
        cap = _env_float("BUSY_BAMIX_V6319_PRELIMITER_PREP_EXTRA_PUSH_CAP_DB", cap_default, 0.0, 1.2)
        try:
            plan["extra_push_db"] = round(min(float(plan.get("extra_push_db", 0.0) or 0.0), float(cap)), 3)
        except Exception:
            plan["extra_push_db"] = 0.0
        plan["bamix_handoff_mode"] = handoff_mode
        plan["bamix_readiness_aware_prep_cap_db"] = round(float(cap), 3)
        plan["policy"] = "BAMix density-completion allows bounded pre-limiter prep so mastering can finish commercial density without reverting to raw-stereo rescue behavior."
    d["commercial_loudness_prep"] = plan
    return d, plan


def _apply_commercial_loudness_prep(y: np.ndarray, sr: int, decision: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    plan = decision.get("commercial_loudness_prep", {}) if isinstance(decision, dict) else {}
    if not isinstance(plan, dict) or not plan.get("active"):
        return y, {"active": False, "reason": plan.get("reason", "not_planned") if isinstance(plan, dict) else "not_planned"}
    out = y
    static_moves = plan.get("static_moves", []) or []
    dynamic_moves = plan.get("dynamic_moves", []) or []
    if static_moves:
        out = apply_eq_moves(out, sr, static_moves, scale=1.0)
    if dynamic_moves:
        # Intensity is already baked into range/gain values; keep dynamic_eq neutral.
        out = dynamic_eq(out, sr, dynamic_moves, intensity=1.0)
    # Re-anchor sub/low stereo before the final push. This is intentionally gentle.
    cleanup_strength = float(plan.get("cleanup_strength", 1.0) or 1.0)
    out = ms_low_mono(out, sr, cutoff_hz=120.0, amount=float(np.clip(0.55 + cleanup_strength * 0.20, 0.55, 0.82)))
    drive = float(plan.get("micro_clip_drive_db", 0.0) or 0.0)
    mix = float(plan.get("micro_clip_mix", 0.0) or 0.0)
    if drive > 0 and mix > 0:
        out = soft_clip(out, drive_db=drive, mix=mix)
    report = {k: v for k, v in plan.items() if k not in {"static_moves", "dynamic_moves"}}
    report.update({
        "active": True,
        "static_move_count": len(static_moves),
        "dynamic_move_count": len(dynamic_moves),
        "applied_micro_clip": bool(drive > 0 and mix > 0),
    })
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report


def _v58_num(value: Any, default: float | None = None) -> float | None:
    try:
        v = float(value)
        if np.isfinite(v):
            return float(v)
    except Exception:
        pass
    return default


def _v58_nested(d: dict[str, Any] | None, *path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _v58_band_db(analysis: dict[str, Any] | None, band: str, default: float = -90.0) -> float:
    if not isinstance(analysis, dict):
        return float(default)
    bands = analysis.get("band_energy_db") if isinstance(analysis.get("band_energy_db"), dict) else {}
    return float(_v58_num(bands.get(band), default) or default)


def _v58_axis_scores(before: dict[str, Any] | None, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    before = before if isinstance(before, dict) else {}
    qi = before.get("quality_indices") if isinstance(before.get("quality_indices"), dict) else {}
    ms = before.get("ms") if isinstance(before.get("ms"), dict) else {}
    mix = before.get("mix_conditioning_analysis") if isinstance(before.get("mix_conditioning_analysis"), dict) else {}
    mix_raw = mix.get("raw_metrics") if isinstance(mix.get("raw_metrics"), dict) else {}
    artifact = before.get("suno_artifact_analysis") if isinstance(before.get("suno_artifact_analysis"), dict) else {}
    risk_scores = artifact.get("risk_scores") if isinstance(artifact.get("risk_scores"), dict) else {}

    mud = float(_v58_num(qi.get("mud_index_db"), _v58_num(mix_raw.get("mud_score"), 0.0) or 0.0) or 0.0)
    harsh = float(_v58_num(qi.get("harshness_index_db"), _v58_num(mix_raw.get("harshness_score"), 0.0) or 0.0) or 0.0)
    air = float(_v58_num(qi.get("air_index_db"), 0.0) or 0.0)
    bass = float(_v58_num(qi.get("bass_index_db"), 0.0) or 0.0)
    lra = float(_v58_num(before.get("lra_lu"), _v58_nested(before, "loudness", "short_term_lufs", "lra_lu", default=2.5)) or 2.5)
    crest = float(_v58_num(before.get("crest_factor_db"), _v58_nested(before, "loudness", "crest_factor_db", default=10.0)) or 10.0)
    tp = float(_v58_num(before.get("approx_true_peak_dbfs"), _v58_nested(before, "loudness", "approx_true_peak_dbfs", default=-6.0)) or -6.0)
    side_low = float(_v58_num(ms.get("side_ratio_lowband"), 0.0) or 0.0)
    side_high = float(_v58_num(ms.get("side_ratio_8k_14k"), 0.0) or 0.0)
    corr = float(_v58_num(before.get("correlation"), ms.get("correlation_avg") if isinstance(ms, dict) else 0.9) or 0.9)
    artifact_risk = float(_v58_num(artifact.get("overall_risk"), mix_raw.get("artifact_score") if isinstance(mix_raw, dict) else 0.0) or 0.0)
    brittle_top = float(_v58_num(risk_scores.get("brittle_8_12k_top"), 0.0) or 0.0)

    # Reference-free, deliberately conservative source-condition axes.  These are
    # not genre exceptions and they do not create boost-first behavior.
    low_mid_bottleneck = float(np.clip((mud - 5.5) / 5.5, 0.0, 1.0))
    bass_headroom_cost = float(np.clip((bass - 14.0) / 9.0, 0.0, 1.0))
    side_high_instability = float(np.clip((side_high - 0.26) / 0.22, 0.0, 1.0))
    side_low_instability = float(np.clip((side_low - 0.12) / 0.22, 0.0, 1.0))
    lra_fragility = float(np.clip((2.3 - lra) / 2.0, 0.0, 1.0))
    true_peak_pressure = float(np.clip((tp + 6.0) / 5.5, 0.0, 1.0))
    crest_fragility = float(np.clip((9.0 - crest) / 4.0, 0.0, 1.0))
    hf_brittleness = float(np.clip(max(0.0, brittle_top, (harsh + 2.0) / 8.0, artifact_risk * 0.55 + side_high_instability * 0.45), 0.0, 1.0))
    air_deficit = float(np.clip((-10.0 - air) / 8.0, 0.0, 1.0)) if air < -10.0 else 0.0

    return {
        "schema_version": "busy_v58_spectral_condition_axes_v8_5_3_58_1",
        "low_mid_bottleneck": round(low_mid_bottleneck, 4),
        "bass_headroom_cost": round(bass_headroom_cost, 4),
        "side_high_instability": round(side_high_instability, 4),
        "side_low_instability": round(side_low_instability, 4),
        "lra_fragility": round(lra_fragility, 4),
        "true_peak_pressure": round(true_peak_pressure, 4),
        "crest_fragility": round(crest_fragility, 4),
        "hf_brittleness": round(hf_brittleness, 4),
        "air_deficit": round(air_deficit, 4),
        "inputs": {
            "mud_index_db": round(mud, 3),
            "harshness_index_db": round(harsh, 3),
            "air_index_db": round(air, 3),
            "bass_index_db": round(bass, 3),
            "lra_lu": round(lra, 3),
            "crest_factor_db": round(crest, 3),
            "true_peak_dbtp": round(tp, 3),
            "side_ratio_lowband": round(side_low, 5),
            "side_ratio_8k_14k": round(side_high, 5),
            "correlation": round(corr, 4),
            "artifact_risk": round(artifact_risk, 3),
            "brittle_8_12k_top": round(brittle_top, 3),
        },
    }


def _v58_db_to_lin(db: Any) -> float:
    v = _v58_num(db, -120.0)
    if v is None:
        v = -120.0
    return float(10.0 ** (float(np.clip(v, -120.0, 24.0)) / 10.0))


def _v58_lin_to_db(x: float, floor_db: float = -120.0) -> float:
    try:
        x = float(x)
        if x > 0 and np.isfinite(x):
            return float(10.0 * np.log10(max(x, 10.0 ** (floor_db / 10.0))))
    except Exception:
        pass
    return float(floor_db)


def _v58_energy_mean_db(before: dict[str, Any] | None, bands: list[str], default: float = -90.0) -> float:
    vals = []
    for b in bands:
        v = _v58_band_db(before, b, default)
        if np.isfinite(v):
            vals.append(_v58_db_to_lin(v))
    if not vals:
        return float(default)
    return _v58_lin_to_db(float(np.mean(vals)), floor_db=-120.0)


def _v58_vocal_lead_importance(decision: dict[str, Any] | None) -> float:
    decision = decision if isinstance(decision, dict) else {}
    candidates: list[Any] = []
    for path in (
        ("final_decision_context", "role_first_dsp_contract", "role_importance", "vocal_front_focus"),
        ("role_first_dsp_contract", "role_importance", "vocal_front_focus"),
        ("input_analysis", "role_first_dsp_contract", "role_map", "role_importance", "vocal_front_focus"),
    ):
        cur: Any = decision
        for k in path:
            if not isinstance(cur, dict):
                cur = None
                break
            cur = cur.get(k)
        candidates.append(cur)
    txt = _profile_text(decision)
    base = 0.0
    if any(k in txt for k in ["vocal", "kpop", "pop", "rnb", "ballad", "spoken", "worship"]):
        base = 0.55
    vals = [float(_v58_num(v, 0.0) or 0.0) for v in candidates]
    return float(np.clip(max([base] + vals), 0.0, 1.0))


def _v58_1_report_driven_spectral_features(before: dict[str, Any] | None, decision: dict[str, Any] | None, axes: dict[str, Any] | None) -> dict[str, Any]:
    """Conservative research-report feature layer for v58.1.

    This is not a full Glasberg-Moore implementation.  It is a deterministic
    proxy layer that maps the source-conditioning research report into safe
    M9/M12 style decisions using only existing analysis features.  Render moves
    remain small, cut/de-masking only, and vocal/lead bands are protected.
    """
    before = before if isinstance(before, dict) else {}
    axes = axes if isinstance(axes, dict) else {}
    qi = before.get("quality_indices") if isinstance(before.get("quality_indices"), dict) else {}
    mud = float(_v58_num(qi.get("mud_index_db"), _v58_nested(axes, "inputs", "mud_index_db", default=0.0)) or 0.0)
    harsh = float(_v58_num(qi.get("harshness_index_db"), _v58_nested(axes, "inputs", "harshness_index_db", default=0.0)) or 0.0)
    bass_idx = float(_v58_num(qi.get("bass_index_db"), _v58_nested(axes, "inputs", "bass_index_db", default=0.0)) or 0.0)
    low_db = _v58_energy_mean_db(before, ["sub_20_60", "bass_60_120", "lowmid_120_500"], -90.0)
    mid_db = _v58_energy_mean_db(before, ["mid_500_2000", "presence_2000_5000"], -90.0)
    high_db = _v58_energy_mean_db(before, ["sibilance_5000_8000", "air_8000_14000"], -90.0)
    sub_db = _v58_band_db(before, "sub_20_60", -90.0)
    bass_db = _v58_band_db(before, "bass_60_120", -90.0)
    lowmid_db = _v58_band_db(before, "lowmid_120_500", -90.0)
    presence_db = _v58_band_db(before, "presence_2000_5000", -90.0)
    sibilance_db = _v58_band_db(before, "sibilance_5000_8000", -90.0)
    air_db = _v58_band_db(before, "air_8000_14000", -90.0)
    low_to_mid_db = low_db - mid_db
    low_to_presence_db = low_db - presence_db
    high_to_mid_db = high_db - mid_db
    sub_to_presence_db = sub_db - presence_db
    bass_to_presence_db = bass_db - presence_db
    low_mid_masking_score = float(np.clip(max((lowmid_db - presence_db - 12.0) / 9.0, (mud - 6.0) / 6.0), 0.0, 1.0))
    bass_masking_score = float(np.clip(max((bass_to_presence_db - 16.0) / 10.0, (sub_to_presence_db - 17.5) / 11.0, (bass_idx - 15.0) / 9.0), 0.0, 1.0))
    hf_masking_score = float(np.clip(max((sibilance_db - presence_db - 2.0) / 8.0, (high_to_mid_db - 1.0) / 8.0, (harsh + 3.0) / 8.0), 0.0, 1.0))
    elatc_low_heavy_score = float(np.clip(max((low_to_mid_db - 11.0) / 8.0, (low_to_presence_db - 17.0) / 9.0, bass_masking_score * 0.82 + low_mid_masking_score * 0.18), 0.0, 1.0))
    elatc_overbright_score = float(np.clip(max((high_to_mid_db - 4.0) / 8.0, hf_masking_score), 0.0, 1.0))
    vocal_importance = _v58_vocal_lead_importance(decision)
    vocal_protection_active = bool(vocal_importance >= 0.62)
    plmd_masking_score = float(np.clip(max(low_mid_masking_score, bass_masking_score, hf_masking_score * (0.55 if vocal_protection_active else 1.0)), 0.0, 1.0))
    masked_bands: list[dict[str, Any]] = []
    if low_mid_masking_score >= 0.35:
        masked_bands.append({"band": "lowmid_120_500", "score": round(low_mid_masking_score, 4), "reason": "low_mid_masks_presence"})
    if bass_masking_score >= 0.35:
        masked_bands.append({"band": "bass_sub_20_120", "score": round(bass_masking_score, 4), "reason": "bass_sub_headroom_masks_presence"})
    if hf_masking_score >= 0.42 and not vocal_protection_active:
        masked_bands.append({"band": "sibilance_air_5k_14k", "score": round(hf_masking_score, 4), "reason": "hf_edge_masks_clarity"})
    return {
        "schema_version": "busy_v58_1_report_driven_spectral_features",
        "m9_elatc": {
            "active_candidate": bool(elatc_low_heavy_score >= 0.35 or elatc_overbright_score >= 0.45 or float(axes.get("air_deficit", 0.0) or 0.0) >= 0.55),
            "low_heavy_score": round(elatc_low_heavy_score, 4),
            "overbright_score": round(elatc_overbright_score, 4),
            "air_deficit_score": round(float(axes.get("air_deficit", 0.0) or 0.0), 4),
            "low_to_mid_db": round(low_to_mid_db, 3),
            "low_to_presence_db": round(low_to_presence_db, 3),
            "high_to_mid_db": round(high_to_mid_db, 3),
            "policy": "ELATC is cut/de-masking only in v58.1; air deficit remains diagnosis-only and never creates a high-shelf boost.",
        },
        "m12_plmd": {
            "active_candidate": bool(plmd_masking_score >= 0.35),
            "proxy_only": True,
            "plmd_masking_score": round(plmd_masking_score, 4),
            "low_mid_masking_score": round(low_mid_masking_score, 4),
            "bass_masking_score": round(bass_masking_score, 4),
            "hf_masking_score": round(hf_masking_score, 4),
            "masked_bands": masked_bands,
            "policy": "Partial-loudness decongestion uses deterministic ERB-like proxy features in v58.1; full Glasberg-Moore rendering remains deferred.",
        },
        "vocal_lead_protection": {
            "importance": round(vocal_importance, 4),
            "active": vocal_protection_active,
        },
        "band_summary_db": {
            "low_db": round(low_db, 3),
            "mid_db": round(mid_db, 3),
            "high_db": round(high_db, 3),
            "sub_20_60": round(sub_db, 3),
            "bass_60_120": round(bass_db, 3),
            "lowmid_120_500": round(lowmid_db, 3),
            "presence_2000_5000": round(presence_db, 3),
            "sibilance_5000_8000": round(sibilance_db, 3),
            "air_8000_14000": round(air_db, 3),
        },
    }


def _v58_move_key(move: dict[str, Any]) -> tuple[str, str, int]:
    target = str(move.get("target", "stereo") or "stereo").lower()
    typ = str(move.get("type", "bell") or "bell").lower().replace("-", "_")
    freq = float(_v58_num(move.get("freq", move.get("frequency_hz", 0.0)), 0.0) or 0.0)
    # Third-octave-ish bins avoid duplicate cuts from several modules while not
    # collapsing genuinely different bands.
    bin_hz = int(round(freq / max(20.0, freq * 0.16))) if freq > 0 else 0
    return target, typ, bin_hz


def _v58_existing_move_pressure(decision: dict[str, Any] | None, before: dict[str, Any] | None = None) -> dict[tuple[str, str, int], float]:
    decision = decision if isinstance(decision, dict) else {}
    before = before if isinstance(before, dict) else {}
    candidates: list[dict[str, Any]] = []
    for key in ("corrective_eq", "character_eq", "dynamic_eq", "suno_repair_static_eq", "suno_repair_dynamic_eq"):
        val = decision.get(key)
        if isinstance(val, list):
            candidates.extend([m for m in val if isinstance(m, dict)])
    prep = decision.get("commercial_loudness_prep") if isinstance(decision.get("commercial_loudness_prep"), dict) else {}
    for key in ("static_moves", "dynamic_moves"):
        val = prep.get(key)
        if isinstance(val, list):
            candidates.extend([m for m in val if isinstance(m, dict)])
    # v62.9.8: A69 residue pre-clean runs before v58/v59 planning in the normal
    # path after cleaned_before is merged.  Include those moves as existing axis
    # pressure so later source-conditioning modules do not keep cutting the same
    # side-high/low-side bands.
    rpc = before.get("residue_pre_clean_stage") if isinstance(before.get("residue_pre_clean_stage"), dict) else {}
    rlc = rpc.get("residue_light_control") if isinstance(rpc.get("residue_light_control"), dict) else {}
    for key in ("static_moves", "dynamic_moves"):
        val = rlc.get(key)
        if isinstance(val, list):
            candidates.extend([m for m in val if isinstance(m, dict)])
    out: dict[tuple[str, str, int], float] = {}
    for m in candidates:
        gain = abs(float(_v58_num(m.get("gain_db", m.get("range_db", 0.0)), 0.0) or 0.0))
        if gain <= 0:
            continue
        k = _v58_move_key(m)
        out[k] = out.get(k, 0.0) + gain
    return out


def _v58_register_move(
    moves: list[dict[str, Any]],
    move: dict[str, Any],
    pressure: dict[tuple[str, str, int], float],
    governor_events: list[dict[str, Any]],
    cap_db: float,
) -> None:
    k = _v58_move_key(move)
    gain = float(_v58_num(move.get("gain_db", move.get("range_db", 0.0)), 0.0) or 0.0)
    existing = float(pressure.get(k, 0.0) or 0.0)
    allowed = max(0.0, float(cap_db) - existing)
    if allowed <= 0.03:
        governor_events.append({"action": "skip", "reason": "axis_already_controlled", "axis": list(k), "existing_abs_db": round(existing, 3), "requested_gain_db": round(gain, 3)})
        return
    if abs(gain) > allowed:
        old = gain
        gain = -allowed if gain < 0 else allowed
        move = dict(move)
        move["gain_db"] = round(gain, 3)
        if "range_db" in move:
            move["range_db"] = round(gain, 3)
        governor_events.append({"action": "downscale", "reason": "axis_cap", "axis": list(k), "existing_abs_db": round(existing, 3), "old_gain_db": round(old, 3), "new_gain_db": round(gain, 3)})
    moves.append(move)
    pressure[k] = existing + abs(float(gain))


def _v58_spectral_conditioning_plan(before: dict[str, Any], decision: dict[str, Any], mode: str) -> dict[str, Any]:
    enabled = _env_bool("BUSY_V58_SPECTRAL_CONDITIONING", True)
    axes = _v58_axis_scores(before, decision)
    if not enabled:
        return {"schema_version": "busy_v58_spectral_conditioning_v8_5_3_58_1", "enabled": False, "active": False, "reason": "disabled", "axes": axes}

    render_enabled = _env_bool("BUSY_V58_SPECTRAL_CONDITIONING_APPLY", True)
    pressure = _v58_existing_move_pressure(decision, before)
    governor_events: list[dict[str, Any]] = []
    static_moves: list[dict[str, Any]] = []
    dynamic_moves: list[dict[str, Any]] = []
    reasons: list[str] = []
    ax = axes
    low_mid = float(ax.get("low_mid_bottleneck", 0.0) or 0.0)
    bass_cost = float(ax.get("bass_headroom_cost", 0.0) or 0.0)
    side_hi = float(ax.get("side_high_instability", 0.0) or 0.0)
    lra_frag = float(ax.get("lra_fragility", 0.0) or 0.0)
    tp_pressure = float(ax.get("true_peak_pressure", 0.0) or 0.0)
    hf_brittle = float(ax.get("hf_brittleness", 0.0) or 0.0)
    side_low = float(ax.get("side_low_instability", 0.0) or 0.0)
    crest_frag = float(ax.get("crest_fragility", 0.0) or 0.0)
    air_def = float(ax.get("air_deficit", 0.0) or 0.0)
    v58_1_enabled = _env_bool("BUSY_V58_1_REPORT_DRIVEN_SPECTRAL_EFFICIENCY", True)
    v58_1_apply = _env_bool("BUSY_V58_1_REPORT_DRIVEN_SPECTRAL_EFFICIENCY_APPLY", True)
    v58_1_features = _v58_1_report_driven_spectral_features(before, decision, axes) if v58_1_enabled else {"enabled": False, "reason": "disabled"}
    v58_1_actions: list[dict[str, Any]] = []

    # This stage is corrective / bottleneck-removal only.  It does not boost a
    # role just because the role is important.  Prominence remains protection and
    # de-masking first.
    if low_mid >= 0.42:
        strength = min(1.0, low_mid)
        reasons.append("low_mid_masking_bottleneck")
        _v58_register_move(static_moves, {"type": "bell", "freq": 285, "gain_db": round(-0.10 - 0.22 * strength, 3), "q": 0.82, "target": "stereo", "source": "v58_masking_aware_low_mid_decongestion"}, pressure, governor_events, cap_db=0.75)
        _v58_register_move(dynamic_moves, {"type": "bell", "freq": 360, "gain_db": round(-0.18 - 0.36 * strength, 3), "q": 0.95, "target": "mid", "source": "v58_dynamic_low_mid_masking_control"}, pressure, governor_events, cap_db=1.10)

    if bass_cost >= 0.50 and (tp_pressure >= 0.35 or lra_frag >= 0.35):
        strength = min(1.0, 0.55 * bass_cost + 0.45 * max(tp_pressure, lra_frag))
        reasons.append("bass_headroom_cost_before_limiter")
        _v58_register_move(dynamic_moves, {"type": "bell", "freq": 72, "gain_db": round(-0.12 - 0.28 * strength, 3), "q": 0.75, "target": "stereo", "source": "v58_loudness_efficiency_lf_headroom_control"}, pressure, governor_events, cap_db=0.80)
        _v58_register_move(dynamic_moves, {"type": "bell", "freq": 118, "gain_db": round(-0.10 - 0.22 * strength, 3), "q": 0.85, "target": "mid", "source": "v58_bass_body_headroom_smoother"}, pressure, governor_events, cap_db=0.70)

    if side_low >= 0.45:
        strength = min(1.0, side_low)
        reasons.append("side_low_efficiency_loss")
        _v58_register_move(static_moves, {"type": "bell", "freq": 145, "gain_db": round(-0.08 - 0.18 * strength, 3), "q": 0.80, "target": "side", "source": "v58_side_low_efficiency_guard"}, pressure, governor_events, cap_db=0.55)

    if side_hi >= 0.35 or hf_brittle >= 0.45:
        strength = min(1.0, max(side_hi, hf_brittle))
        reasons.append("side_high_or_hf_brittleness_bottleneck")
        _v58_register_move(dynamic_moves, {"type": "bell", "freq": 9200, "gain_db": round(-0.12 - 0.32 * strength, 3), "q": 1.25, "target": "side", "source": "v58_side_high_hash_limiter_stability_control"}, pressure, governor_events, cap_db=0.85)
        if hf_brittle >= 0.55:
            _v58_register_move(dynamic_moves, {"type": "bell", "freq": 11800, "gain_db": round(-0.08 - 0.22 * strength, 3), "q": 1.0, "target": "stereo", "source": "v58_hf_brittleness_dynamic_guard"}, pressure, governor_events, cap_db=0.65)

    v58_1_static_before = len(static_moves)
    v58_1_dynamic_before = len(dynamic_moves)
    if v58_1_enabled:
        m9 = v58_1_features.get("m9_elatc", {}) if isinstance(v58_1_features, dict) else {}
        m12 = v58_1_features.get("m12_plmd", {}) if isinstance(v58_1_features, dict) else {}
        vocal_protect = bool(_v58_nested(v58_1_features, "vocal_lead_protection", "active", default=False)) if isinstance(v58_1_features, dict) else False
        m9_low = float(m9.get("low_heavy_score", 0.0) or 0.0) if isinstance(m9, dict) else 0.0
        m9_bright = float(m9.get("overbright_score", 0.0) or 0.0) if isinstance(m9, dict) else 0.0
        m12_score = float(m12.get("plmd_masking_score", 0.0) or 0.0) if isinstance(m12, dict) else 0.0
        if m9_low >= 0.45 and (tp_pressure >= 0.25 or bass_cost >= 0.35 or low_mid >= 0.35):
            strength = min(1.0, 0.60 * m9_low + 0.20 * max(tp_pressure, bass_cost) + 0.20 * low_mid)
            reasons.append("v58_1_m9_equal_loudness_low_heavy_tilt")
            v58_1_actions.append({"module": "M9_ELATC", "action": "low_heavy_tilt_cut", "strength": round(strength, 4)})
            if v58_1_apply:
                _v58_register_move(static_moves, {"type": "low_shelf", "freq": 155, "gain_db": round(-0.08 - 0.22 * strength, 3), "q": 0.72, "target": "stereo", "source": "v58_1_m9_elatc_low_heavy_efficiency_tilt"}, pressure, governor_events, cap_db=0.55)
                _v58_register_move(static_moves, {"type": "bell", "freq": 430, "gain_db": round(-0.06 - 0.16 * strength, 3), "q": 0.74, "target": "stereo", "source": "v58_1_m9_elatc_low_mid_tilt_trim"}, pressure, governor_events, cap_db=0.45)
        if m9_bright >= 0.58 and hf_brittle >= 0.40:
            strength = min(1.0, 0.65 * m9_bright + 0.35 * hf_brittle)
            reasons.append("v58_1_m9_equal_loudness_overbright_guard")
            v58_1_actions.append({"module": "M9_ELATC", "action": "overbright_guard_cut", "strength": round(strength, 4)})
            if v58_1_apply:
                _v58_register_move(dynamic_moves, {"type": "high_shelf", "freq": 7600, "gain_db": round(-0.06 - 0.20 * strength, 3), "q": 0.75, "target": "stereo", "source": "v58_1_m9_elatc_overbright_efficiency_guard"}, pressure, governor_events, cap_db=0.50)
        if m12_score >= 0.42:
            strength = min(1.0, m12_score)
            reasons.append("v58_1_m12_partial_loudness_masking_decongestion_proxy")
            v58_1_actions.append({"module": "M12_PLMD", "action": "masked_energy_decongestion_proxy", "strength": round(strength, 4), "masked_bands": m12.get("masked_bands", []) if isinstance(m12, dict) else []})
            masked_names = [str(b.get("band", "")) for b in (m12.get("masked_bands", []) if isinstance(m12, dict) else []) if isinstance(b, dict)]
            if v58_1_apply:
                if any("lowmid" in b for b in masked_names):
                    _v58_register_move(dynamic_moves, {"type": "bell", "freq": 520, "gain_db": round(-0.10 - 0.28 * strength, 3), "q": 1.05, "target": "mid", "source": "v58_1_m12_plmd_lowmid_masking_proxy"}, pressure, governor_events, cap_db=0.70)
                if any("bass_sub" in b for b in masked_names):
                    _v58_register_move(dynamic_moves, {"type": "bell", "freq": 96, "gain_db": round(-0.08 - 0.24 * strength, 3), "q": 0.90, "target": "stereo", "source": "v58_1_m12_plmd_bass_masking_proxy"}, pressure, governor_events, cap_db=0.55)
                if any("sibilance" in b for b in masked_names) and not vocal_protect:
                    _v58_register_move(dynamic_moves, {"type": "bell", "freq": 6500, "gain_db": round(-0.06 - 0.18 * strength, 3), "q": 1.20, "target": "stereo", "source": "v58_1_m12_plmd_hf_masking_proxy"}, pressure, governor_events, cap_db=0.45)
        if not v58_1_apply and (m9_low >= 0.45 or m9_bright >= 0.58 or m12_score >= 0.42):
            reasons.append("v58_1_report_driven_spectral_efficiency_analysis_only")
            governor_events.append({"action": "v58_1_analysis_only", "reason": "BUSY_V58_1_REPORT_DRIVEN_SPECTRAL_EFFICIENCY_APPLY=0"})

    if air_def >= 0.65:
        # Diagnosis only for now.  Boosting 'air' on brittle AI masters often worsens
        # fizz and limiter fatigue; v58/v58.1 records this as a preservation/de-masking hint.
        reasons.append("air_deficit_diagnosis_only_no_boost")
        if v58_1_enabled:
            v58_1_actions.append({"module": "M9_ELATC", "action": "air_deficit_recorded_no_boost", "strength": round(float(air_def), 4)})

    if lra_frag >= 0.55 or crest_frag >= 0.45:
        reasons.append("lra_or_crest_fragility_limits_eq_strength")
        # Further reduce all v58 render-affecting EQ when the source is already
        # fragile.  This is a damage prevention step, not a loudness push.
        scale = float(np.clip(1.0 - 0.35 * max(lra_frag, crest_frag), 0.55, 1.0))
        for m in static_moves + dynamic_moves:
            try:
                m["gain_db"] = round(float(m.get("gain_db", 0.0)) * scale, 3)
            except Exception:
                pass
        governor_events.append({"action": "fragility_scale", "scale": round(scale, 3), "reason": "lra_or_crest_fragility"})

    v58_1_applied_static_move_count = max(0, len(static_moves) - int(v58_1_static_before))
    v58_1_applied_dynamic_move_count = max(0, len(dynamic_moves) - int(v58_1_dynamic_before))
    v58_1_applied_move_count = v58_1_applied_static_move_count + v58_1_applied_dynamic_move_count
    active = bool(static_moves or dynamic_moves or reasons)
    return {
        "schema_version": "busy_v58_spectral_conditioning_v8_5_3_58_1",
        "enabled": True,
        "active": bool(active and render_enabled),
        "diagnosis_active": bool(active),
        "render_affecting": bool(render_enabled and (static_moves or dynamic_moves)),
        "mode": "render" if render_enabled else "analysis_only",
        "reasons": reasons,
        "axes": axes,
        "static_moves": static_moves,
        "dynamic_moves": dynamic_moves,
        "static_move_count": len(static_moves),
        "dynamic_move_count": len(dynamic_moves),
        "v58_1_report_driven_spectral_efficiency": {
            "schema_version": "busy_v58_1_report_driven_spectral_efficiency",
            "enabled": bool(v58_1_enabled),
            "apply_enabled": bool(v58_1_apply),
            "active": bool(v58_1_enabled and v58_1_actions),
            "render_affecting": bool(v58_1_enabled and v58_1_apply and v58_1_applied_move_count > 0),
            "applied_static_move_count": int(v58_1_applied_static_move_count),
            "applied_dynamic_move_count": int(v58_1_applied_dynamic_move_count),
            "applied_move_count": int(v58_1_applied_move_count),
            "features": v58_1_features,
            "actions": v58_1_actions,
            "policy": "M9 ELATC and M12 PLMD are report-driven source-conditioning modules in v58.1. They are conservative cut/de-masking only; no air boost and no target-LUFS command.",
        },
        "axis_ownership_ledger": _axis_ledger_from_pressure(pressure),
        "eq_dynamic_stack_governor": {
            "schema_version": "busy_v58_eq_dynamic_stack_governor_v8_5_3_62_9_8",
            "events": governor_events,
            "policy": "v58 moves are capped by correction axis so corrective EQ, dynamic EQ, commercial prep and IVRCF/direct moves do not stack blindly.",
        },
        "policy": "Masking-aware spectral bottleneck conditioning before loudness push. Corrective/de-masking cuts only; no role boost and no target-LUFS command.",
    }


def _attach_v58_spectral_conditioning(before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    d = copy.deepcopy(decision or {})
    plan = _v58_spectral_conditioning_plan(before, d, mode)
    d["v58_spectral_conditioning_plan"] = plan
    return d, plan


def _apply_v58_spectral_conditioning(y: np.ndarray, sr: int, decision: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    plan = decision.get("v58_spectral_conditioning_plan", {}) if isinstance(decision, dict) else {}
    if not isinstance(plan, dict) or not plan.get("enabled"):
        return y, {"enabled": False, "active": False, "reason": plan.get("reason", "not_planned") if isinstance(plan, dict) else "not_planned"}
    if not plan.get("active"):
        report = {k: v for k, v in plan.items() if k not in {"static_moves", "dynamic_moves"}}
        report.update({"active": False, "reason": "analysis_only_or_no_render_moves"})
        return y, report
    out = y
    static_moves = [m for m in (plan.get("static_moves", []) or []) if isinstance(m, dict)]
    dynamic_moves = [m for m in (plan.get("dynamic_moves", []) or []) if isinstance(m, dict)]
    if static_moves:
        out = apply_eq_moves(out, sr, static_moves, scale=1.0)
    if dynamic_moves:
        out = dynamic_eq(out, sr, dynamic_moves, intensity=1.0)
    report = {k: v for k, v in plan.items() if k not in {"static_moves", "dynamic_moves"}}
    report.update({
        "active": True,
        "applied_static_move_count": len(static_moves),
        "applied_dynamic_move_count": len(dynamic_moves),
    })
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report


def _v59_lf_crest_source_conditioning_plan(before: dict[str, Any] | None, decision: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    """v59: report-driven LF/crest/rogue-peak source-conditioning plan.

    This is the first report-derived dynamics/source-conditioning layer after
    v58.1.  It is intentionally conservative: it never chases a target LUFS,
    never boosts prominence, and it reduces action strength when LRA/crest are
    already fragile.  M3/M4 are time-domain peak hygiene; M7/M8 are LF/headroom
    dynamic controls.
    """
    axes = _v58_axis_scores(before if isinstance(before, dict) else {}, decision if isinstance(decision, dict) else {})
    if not _env_bool("BUSY_V59_LF_CREST_SOURCE_CONDITIONING", True):
        return {"schema_version": "busy_v59_lf_crest_source_conditioning_v8_5_3_59_2", "enabled": False, "active": False, "reason": "disabled", "axes": axes}
    before = before if isinstance(before, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    apply_enabled = _env_bool("BUSY_V59_LF_CREST_SOURCE_CONDITIONING_APPLY", True)
    pressure = _v58_existing_move_pressure(decision, before if isinstance(before, dict) else {})
    governor_events: list[dict[str, Any]] = []
    dynamic_moves: list[dict[str, Any]] = []
    static_moves: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    reasons: list[str] = []

    inp = axes.get("inputs", {}) if isinstance(axes.get("inputs"), dict) else {}
    lra = float(_v58_num(inp.get("lra_lu"), 2.5) or 2.5)
    crest = float(_v58_num(inp.get("crest_factor_db"), 10.0) or 10.0)
    tp = float(_v58_num(inp.get("true_peak_dbtp"), -6.0) or -6.0)
    lra_fragility = float(_v58_num(axes.get("lra_fragility"), 0.0) or 0.0)
    crest_fragility = float(_v58_num(axes.get("crest_fragility"), 0.0) or 0.0)
    tp_pressure = float(_v58_num(axes.get("true_peak_pressure"), 0.0) or 0.0)
    bass_cost = float(_v58_num(axes.get("bass_headroom_cost"), 0.0) or 0.0)
    low_mid = float(_v58_num(axes.get("low_mid_bottleneck"), 0.0) or 0.0)
    v6299 = decision.get("v62_9_9_research_governance") if isinstance(decision.get("v62_9_9_research_governance"), dict) else {}
    v6299_micro = v6299.get("microdynamics_lra_shape_recovery") if isinstance(v6299.get("microdynamics_lra_shape_recovery"), dict) else {}
    v6299_low = v6299.get("translation_aware_low_end_governor") if isinstance(v6299.get("translation_aware_low_end_governor"), dict) else {}
    v6299_dyn_scale = float(np.clip(float(v6299_micro.get("dynamic_eq_intensity_scale", 1.0) or 1.0), 0.55, 1.05))
    v6299_mb_scale = float(np.clip(float(v6299_micro.get("multiband_amount_scale", 1.0) or 1.0), 0.50, 1.05))
    v6299_upward_scale = float(np.clip(float(v6299_micro.get("upward_density_scale", 1.0) or 1.0), 0.25, 1.05))
    v6299_low_scale = float(np.clip(float(v6299_low.get("low_end_control_scale", 1.0) or 1.0), 0.72, 1.16))

    sub_db = _v58_band_db(before, "sub_20_60", -90.0)
    bass_db = _v58_band_db(before, "bass_60_120", -90.0)
    lowmid_db = _v58_band_db(before, "lowmid_120_500", -90.0)
    presence_db = _v58_band_db(before, "presence_2000_5000", -90.0)
    air_db = _v58_band_db(before, "air_8000_14000", -90.0)

    # Fragility scaling: if the track is already flat, v59 can still remove rogue
    # peaks/LF waste, but it must not behave like another compressor.
    fragility_scale = float(np.clip(1.0 - 0.58 * lra_fragility - 0.28 * crest_fragility, 0.18, 1.0))
    if lra < 1.25:
        fragility_scale = min(fragility_scale, 0.42)
    if crest < 8.5:
        fragility_scale = min(fragility_scale, 0.55)

    # M7: sub/LF envelope compaction proxy.  It creates dynamic LF cuts only
    # where LF headroom cost is high and scales down heavily on flat material.
    lf_excess_vs_presence = float(np.clip(((max(sub_db, bass_db) - presence_db) - 16.0) / 16.0, 0.0, 1.0)) if presence_db > -89 else 0.0
    m7_score = float(np.clip(max(bass_cost, 0.65 * lf_excess_vs_presence + 0.35 * tp_pressure), 0.0, 1.0))
    if m7_score >= 0.34:
        strength = float(np.clip(m7_score * fragility_scale * v6299_low_scale, 0.0, 0.75))
        actions.append({"module": "M7_SLFEC", "action": "sub_band_lf_envelope_compaction_proxy", "strength": round(strength, 4), "raw_score": round(m7_score, 4)})
        reasons.append("v59_m7_lf_envelope_headroom_cost")
        if apply_enabled and strength >= 0.08:
            _v58_register_move(dynamic_moves, {"type": "bell", "freq": 46, "gain_db": round(-0.10 - 0.32 * strength, 3), "q": 0.78, "target": "mid", "source": "v59_m7_sub_lf_envelope_compactor"}, pressure, governor_events, cap_db=0.70)
            _v58_register_move(dynamic_moves, {"type": "bell", "freq": 82, "gain_db": round(-0.08 - 0.28 * strength, 3), "q": 0.86, "target": "stereo", "source": "v59_m7_bass_envelope_headroom_compactor"}, pressure, governor_events, cap_db=0.62)

    # M8: LF note/crest stabilization proxy.  Without pitch tracking, use a very
    # conservative dominance test from band energy relationships.
    lf_dominance = float(np.clip(((bass_db - lowmid_db) - 5.0) / 11.0, 0.0, 1.0)) if lowmid_db > -89 else 0.0
    sub_dominance = float(np.clip(((sub_db - bass_db) - 3.0) / 10.0, 0.0, 1.0)) if bass_db > -89 else 0.0
    m8_score = float(np.clip(max(lf_dominance, sub_dominance) * (0.55 + 0.45 * tp_pressure), 0.0, 1.0))
    if m8_score >= 0.34:
        strength = float(np.clip(m8_score * fragility_scale * v6299_low_scale, 0.0, 0.70))
        actions.append({"module": "M8_LFCS", "action": "lf_crest_stabilizer_proxy", "strength": round(strength, 4), "raw_score": round(m8_score, 4), "lf_dominance": round(lf_dominance, 4), "sub_dominance": round(sub_dominance, 4)})
        reasons.append("v59_m8_lf_crest_dominant_band")
        if apply_enabled and strength >= 0.08:
            center = 58 if sub_dominance > lf_dominance else 112
            _v58_register_move(dynamic_moves, {"type": "bell", "freq": center, "gain_db": round(-0.08 - 0.30 * strength, 3), "q": 1.35, "target": "stereo", "source": "v59_m8_lf_crest_stabilizer"}, pressure, governor_events, cap_db=0.65)

    # M3/M4: time-domain rogue spike/outlier hygiene.  This is not a limiter and
    # not target chasing; it only tamps extreme short outliers before the normal
    # chain.  On flat/damaged material the maximum GR is smaller.
    peak_hygiene_score = float(np.clip(max(tp_pressure, (crest - 10.5) / 4.5, 0.0) * (0.65 + 0.35 * bass_cost), 0.0, 1.0))
    time_domain_actions: list[dict[str, Any]] = []
    if peak_hygiene_score >= 0.42:
        max_gr = float(np.clip(0.35 + 0.95 * peak_hygiene_score, 0.35, 1.20))
        if lra < 1.25 or crest < 8.5:
            max_gr = min(max_gr, 0.55)
        pct = 99.88 if peak_hygiene_score < 0.70 else 99.82
        action = {"module": "M3_M4", "action": "rogue_spike_outlier_tamer", "strength": round(peak_hygiene_score, 4), "percentile": pct, "max_gr_db": round(max_gr, 3)}
        actions.append(action)
        time_domain_actions.append(action)
        reasons.append("v59_m3_m4_rogue_peak_outlier_hygiene")

    # v59.2: program-dependent dynamics and stereo/side energy conditioning.
    # This layer does not chase target LUFS.  It reduces stacking/compression
    # pressure when LRA/crest are already fragile and adds very small M10/M11
    # stereo-side stabilizers only when measured stereo axes justify them.
    v59_2_enabled = _env_bool("BUSY_V59_2_PROGRAM_DYNAMICS_STEREO_SIDE", True)
    v59_2_apply = bool(v59_2_enabled and _env_bool("BUSY_V59_2_PROGRAM_DYNAMICS_STEREO_SIDE_APPLY", True) and apply_enabled)
    side_high = float(_v58_num(axes.get("side_high_instability"), 0.0) or 0.0)
    side_low = float(_v58_num(axes.get("side_low_instability"), 0.0) or 0.0)
    corr = float(_v58_num(inp.get("correlation"), 0.92) or 0.92)
    corr_risk = float(np.clip((0.94 - corr) / 0.22, 0.0, 1.0))
    # Preserve microdynamics first on very flat material.  These controls are
    # consumed by _render_once to scale the existing dynamic EQ/MBC/upward-density
    # stages rather than adding another compressor.
    mb_scale = float(np.clip((1.0 - 0.34 * lra_fragility - 0.18 * crest_fragility) * v6299_mb_scale, 0.50, 1.0))
    dyn_eq_scale = float(np.clip((1.0 - 0.28 * lra_fragility - 0.12 * crest_fragility) * v6299_dyn_scale, 0.58, 1.0))
    upward_density_scale = float(np.clip((1.0 - 0.55 * lra_fragility - 0.20 * crest_fragility) * v6299_upward_scale, 0.25, 1.0))
    if lra < 1.15:
        mb_scale = min(mb_scale, 0.72)
        dyn_eq_scale = min(dyn_eq_scale, 0.78)
        upward_density_scale = min(upward_density_scale, 0.44)
    v59_2_actions: list[dict[str, Any]] = []
    stereo_side_actions: list[dict[str, Any]] = []
    v59_2_reasons: list[str] = []
    if v59_2_enabled:
        if mb_scale < 0.94 or dyn_eq_scale < 0.94 or upward_density_scale < 0.94:
            v59_2_reasons.append("v59_2_program_dependent_dynamics_lra_crest_preservation")
            v59_2_actions.append({
                "module": "V59_2_PDD",
                "action": "program_dependent_dynamics_scale",
                "mb_amount_scale": round(mb_scale, 4),
                "dynamic_eq_intensity_scale": round(dyn_eq_scale, 4),
                "upward_density_scale": round(upward_density_scale, 4),
                "reason": "lra_or_crest_fragility",
            })
        m10_score = float(np.clip(max(corr_risk * 0.72, side_low * 0.45 + corr_risk * 0.35), 0.0, 1.0))
        if m10_score >= 0.30:
            strength = float(np.clip(m10_score * (0.65 + 0.35 * min(1.0, lra_fragility)), 0.0, 0.75))
            action = {"module": "M10_ALREB", "action": "asymmetric_lr_energy_balancer", "strength": round(strength, 4), "max_gain_db": round(0.18 + 0.32 * strength, 3), "raw_score": round(m10_score, 4)}
            actions.append(action); v59_2_actions.append(action); stereo_side_actions.append(action)
            v59_2_reasons.append("v59_2_m10_asymmetric_lr_energy_balance")
        m11_score = float(np.clip(max(side_high, 0.55 * side_high + 0.30 * corr_risk + 0.15 * tp_pressure), 0.0, 1.0))
        if m11_score >= 0.28:
            strength = float(np.clip(m11_score * (0.75 + 0.25 * tp_pressure), 0.0, 0.80))
            action = {"module": "M11_SBST", "action": "side_band_spike_tamer", "strength": round(strength, 4), "max_gr_db": round(0.35 + 1.10 * strength, 3), "raw_score": round(m11_score, 4), "side_high_instability": round(side_high, 4)}
            actions.append(action); v59_2_actions.append(action); stereo_side_actions.append(action)
            v59_2_reasons.append("v59_2_m11_side_band_spike_tamer")
        reasons.extend([r for r in v59_2_reasons if r not in reasons])
        if not v59_2_apply and (v59_2_actions or stereo_side_actions):
            governor_events.append({"action": "v59_2_analysis_only", "reason": "BUSY_V59_2_PROGRAM_DYNAMICS_STEREO_SIDE_APPLY=0 or parent apply disabled"})

    if not apply_enabled and (dynamic_moves or time_domain_actions or stereo_side_actions):
        governor_events.append({"action": "v59_analysis_only", "reason": "BUSY_V59_LF_CREST_SOURCE_CONDITIONING_APPLY=0"})

    active = bool(actions or (v59_2_enabled and v59_2_actions))
    render_affecting = bool(apply_enabled and (dynamic_moves or time_domain_actions or (v59_2_apply and stereo_side_actions)))
    return {
        "schema_version": "busy_v59_lf_crest_source_conditioning_v8_5_3_59_2",
        "enabled": True,
        "apply_enabled": bool(apply_enabled),
        "active": active,
        "render_affecting": render_affecting,
        "axes": axes,
        "features": {
            "sub_db": round(sub_db, 3), "bass_db": round(bass_db, 3), "lowmid_db": round(lowmid_db, 3), "presence_db": round(presence_db, 3), "air_db": round(air_db, 3),
            "lra_lu": round(lra, 3), "crest_factor_db": round(crest, 3), "true_peak_dbtp": round(tp, 3),
            "fragility_scale": round(fragility_scale, 4), "lf_excess_vs_presence": round(lf_excess_vs_presence, 4),
            "v6299_dynamic_eq_scale": round(v6299_dyn_scale, 4), "v6299_multiband_scale": round(v6299_mb_scale, 4),
            "v6299_upward_density_scale": round(v6299_upward_scale, 4), "v6299_low_end_control_scale": round(v6299_low_scale, 4),
        },
        "v62_9_9_research_governance_inputs": {
            "microdynamics_lra_shape_recovery": v6299_micro,
            "translation_aware_low_end_governor": v6299_low,
        },
        "actions": actions,
        "reasons": reasons,
        "v59_2_program_dependent_dynamics_stereo_side": {
            "schema_version": "busy_v59_2_program_dependent_dynamics_stereo_side",
            "enabled": bool(v59_2_enabled),
            "apply_enabled": bool(v59_2_apply),
            "active": bool(v59_2_enabled and v59_2_actions),
            "render_affecting": bool(v59_2_apply and stereo_side_actions),
            "reasons": v59_2_reasons,
            "actions": v59_2_actions,
            "stereo_side_actions": stereo_side_actions,
            "program_dynamics_controls": {
                "multiband_amount_scale": round(mb_scale, 4),
                "dynamic_eq_intensity_scale": round(dyn_eq_scale, 4),
                "upward_density_scale": round(upward_density_scale, 4),
                "policy": "Preserve LRA/microdynamics by scaling existing dynamics stages; does not create loudness push.",
            },
            "features": {"side_high_instability": round(side_high, 4), "side_low_instability": round(side_low, 4), "correlation": round(corr, 4), "corr_risk": round(corr_risk, 4), "lra_fragility": round(lra_fragility, 4), "crest_fragility": round(crest_fragility, 4)},
            "policy": "M10/M11 are very small stereo/side stabilizers; program-dependent dynamics scales existing compression instead of adding new compression.",
        },
        "stereo_side_actions": stereo_side_actions,
        "stereo_side_action_count": len(stereo_side_actions),
        "static_moves": static_moves,
        "dynamic_moves": dynamic_moves,
        "time_domain_actions": time_domain_actions,
        "static_move_count": len(static_moves),
        "dynamic_move_count": len(dynamic_moves),
        "time_domain_action_count": len(time_domain_actions),
        "axis_ownership_ledger": _axis_ledger_from_pressure(pressure),
        "governor": {"events": governor_events, "policy": "v59 moves pass through v58 EQ/DynamicEQ stack governor and LRA/crest fragility scaling."},
        "policy": "M3/M4/M7/M8 are conservative source-conditioning modules: cut/peak-hygiene only, no target-LUFS command, no prominence boost.",
    }


def _attach_v59_lf_crest_source_conditioning(before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    d = copy.deepcopy(decision or {})
    plan = _v59_lf_crest_source_conditioning_plan(before, d, mode)
    d["v59_lf_crest_source_conditioning_plan"] = plan
    return d, plan


def _v59_apply_rogue_peak_outlier_tamer(y: np.ndarray, action: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        arr = np.asarray(y, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        mono_abs = np.max(np.abs(arr), axis=1)
        finite = mono_abs[np.isfinite(mono_abs)]
        finite = finite[finite > 1e-9]
        if finite.size < 2048:
            return y, {"active": False, "reason": "too_few_samples"}
        pct = float(action.get("percentile", 99.88) or 99.88)
        pct = float(np.clip(pct, 99.50, 99.98))
        thr = float(np.percentile(finite, pct))
        peak = float(np.max(finite))
        if not np.isfinite(thr) or thr <= 1e-8 or peak <= thr * 1.01:
            return y, {"active": False, "reason": "no_extreme_outliers", "threshold": round(thr, 8), "peak": round(peak, 8)}
        max_gr_db = float(np.clip(float(action.get("max_gr_db", 0.65) or 0.65), 0.15, 1.25))
        ratio = 0.38
        mask = mono_abs > thr
        if int(np.sum(mask)) < 3:
            return y, {"active": False, "reason": "outlier_count_too_low", "threshold": round(thr, 8), "peak": round(peak, 8)}
        target = thr + (mono_abs - thr) * ratio
        gain_floor = 10.0 ** (-max_gr_db / 20.0)
        target = np.maximum(target, mono_abs * gain_floor)
        scale = np.ones_like(mono_abs)
        scale[mask] = np.clip(target[mask] / np.maximum(mono_abs[mask], 1e-12), gain_floor, 1.0)
        # Smooth the scale curve over a few samples to avoid zippering, but keep it
        # short enough that it does not act as a compressor envelope.
        try:
            kernel = np.ones(5, dtype=np.float64) / 5.0
            scale = np.minimum(1.0, np.convolve(scale, kernel, mode="same"))
        except Exception:
            pass
        out = arr * scale[:, None]
        gr_db = -20.0 * np.log10(np.maximum(scale[mask], 1e-12)) if np.any(mask) else np.asarray([], dtype=np.float64)
        report = {
            "active": True,
            "module": action.get("module", "M3_M4"),
            "action": action.get("action", "rogue_spike_outlier_tamer"),
            "percentile": pct,
            "threshold_abs": round(thr, 8),
            "peak_abs_before": round(peak, 8),
            "peak_abs_after": round(float(np.max(np.max(np.abs(out), axis=1))), 8),
            "event_sample_count": int(np.sum(mask)),
            "event_ratio": round(float(np.mean(mask)), 8),
            "max_gr_db": round(float(np.max(gr_db)) if gr_db.size else 0.0, 4),
            "mean_gr_db_on_events": round(float(np.mean(gr_db)) if gr_db.size else 0.0, 4),
            "policy": "Sub-ms/short outlier hygiene only; bounded GR and no target-loudness chasing.",
        }
        if y.ndim == 1:
            out = out[:, 0]
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report
    except Exception as exc:
        return y, {"active": False, "reason": "error", "error_type": type(exc).__name__, "error": str(exc)[:220]}



def _v59_apply_stereo_side_action(y: np.ndarray, sr: int, action: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply v59.2 M10/M11 stereo-side stabilizers conservatively."""
    try:
        arr = np.asarray(y, dtype=np.float64)
        mono_in = arr.ndim == 1
        if mono_in:
            arr = arr[:, None]
        if arr.shape[1] < 2 or arr.shape[0] < max(512, int(sr * 0.05)):
            return y, {"active": False, "reason": "mono_or_too_short", "module": action.get("module")}
        module = str(action.get("module") or "")
        out = arr.copy()
        if module == "M10_ALREB":
            # Slow global L/R energy balancing.  This intentionally ignores tiny
            # differences and only trims the louder side by a bounded amount.
            rms_l = float(np.sqrt(np.mean(out[:, 0] ** 2) + 1e-12))
            rms_r = float(np.sqrt(np.mean(out[:, 1] ** 2) + 1e-12))
            diff_db = 20.0 * np.log10(max(rms_l, 1e-12) / max(rms_r, 1e-12))
            max_gain_db = float(np.clip(float(action.get("max_gain_db", 0.35) or 0.35), 0.10, 0.50))
            if abs(diff_db) < 0.55:
                return y, {"active": False, "module": module, "action": action.get("action"), "reason": "lr_imbalance_below_gate", "lr_diff_db": round(float(diff_db), 4)}
            trim_db = -float(np.clip(abs(diff_db) * 0.18, 0.05, max_gain_db))
            if diff_db > 0:
                out[:, 0] *= 10.0 ** (trim_db / 20.0)
                gain_l, gain_r = trim_db, 0.0
            else:
                out[:, 1] *= 10.0 ** (trim_db / 20.0)
                gain_l, gain_r = 0.0, trim_db
            if mono_in:
                out = out[:, 0]
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), {"active": True, "module": module, "action": action.get("action"), "lr_diff_db_before": round(float(diff_db), 4), "gain_l_db": round(float(gain_l), 4), "gain_r_db": round(float(gain_r), 4), "policy": "Bounded L/R energy trim only; no widening or boost."}
        if module == "M11_SBST":
            mid, side = lr_to_ms(out)
            mid_env = lfilter([0.025], [1.0, -0.975], np.abs(mid))
            side_env = lfilter([0.025], [1.0, -0.975], np.abs(side))
            ratio = side_env / np.maximum(mid_env, 1e-9)
            # Adaptive threshold: catches short side excursions, not stable width.
            finite = ratio[np.isfinite(ratio)]
            if finite.size < 1024:
                return y, {"active": False, "module": module, "action": action.get("action"), "reason": "too_few_ratio_samples"}
            threshold = float(np.clip(np.percentile(finite, 88.0), 0.42, 1.35))
            over = np.clip((ratio - threshold) / max(threshold, 1e-9), 0.0, 1.0)
            if float(np.mean(over > 0.02)) < 0.002:
                return y, {"active": False, "module": module, "action": action.get("action"), "reason": "side_spike_density_below_gate", "threshold": round(threshold, 4)}
            max_gr_db = float(np.clip(float(action.get("max_gr_db", 0.75) or 0.75), 0.20, 1.50))
            strength = float(np.clip(float(action.get("strength", 0.4) or 0.4), 0.0, 1.0))
            gr_db = -max_gr_db * strength * over
            # Smooth the gain curve so it behaves like a side-spike tamer, not a tremolo.
            gain = 10.0 ** (gr_db / 20.0)
            try:
                gain = lfilter([0.035], [1.0, -0.965], gain)
                gain = np.clip(gain, 10.0 ** (-max_gr_db / 20.0), 1.0)
            except Exception:
                pass
            side2 = side * gain
            out = ms_to_lr(mid, side2)
            active_ratio = float(np.mean(over > 0.02))
            if mono_in:
                out = out[:, 0]
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), {"active": True, "module": module, "action": action.get("action"), "threshold": round(threshold, 4), "active_ratio": round(active_ratio, 6), "max_gr_db": round(float(-np.min(gr_db)) if gr_db.size else 0.0, 4), "policy": "Side-only spike attenuation; mono-sum and intentional width protected by conservative gates."}
        return y, {"active": False, "module": module, "action": action.get("action"), "reason": "unsupported_v59_2_action"}
    except Exception as exc:
        return y, {"active": False, "module": action.get("module"), "action": action.get("action"), "reason": "error", "error_type": type(exc).__name__, "error": str(exc)[:240]}

def _apply_v59_lf_crest_source_conditioning(y: np.ndarray, sr: int, decision: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    plan = decision.get("v59_lf_crest_source_conditioning_plan", {}) if isinstance(decision, dict) else {}
    if not isinstance(plan, dict) or not plan.get("enabled"):
        return y, {"enabled": False, "active": False, "reason": plan.get("reason", "not_planned") if isinstance(plan, dict) else "not_planned"}
    report = {k: v for k, v in plan.items() if k not in {"static_moves", "dynamic_moves"}}
    if not plan.get("render_affecting"):
        report.update({"active": bool(plan.get("active")), "render_affecting": False, "reason": "analysis_only_or_no_render_moves"})
        return y, report
    out = y
    time_reports: list[dict[str, Any]] = []
    for action in [a for a in (plan.get("time_domain_actions", []) or []) if isinstance(a, dict)]:
        out, tr = _v59_apply_rogue_peak_outlier_tamer(out, action)
        time_reports.append(tr)
    stereo_side_reports: list[dict[str, Any]] = []
    v59_2 = plan.get("v59_2_program_dependent_dynamics_stereo_side") if isinstance(plan.get("v59_2_program_dependent_dynamics_stereo_side"), dict) else {}
    for action in [a for a in (plan.get("stereo_side_actions", []) or v59_2.get("stereo_side_actions", []) or []) if isinstance(a, dict)]:
        out, tr = _v59_apply_stereo_side_action(out, sr, action)
        stereo_side_reports.append(tr)
    static_moves = [m for m in (plan.get("static_moves", []) or []) if isinstance(m, dict)]
    dynamic_moves = [m for m in (plan.get("dynamic_moves", []) or []) if isinstance(m, dict)]
    if static_moves:
        out = apply_eq_moves(out, sr, static_moves, scale=1.0)
    if dynamic_moves:
        out = dynamic_eq(out, sr, dynamic_moves, intensity=1.0)
    report.update({
        "active": True,
        "render_affecting": True,
        "applied_static_move_count": len(static_moves),
        "applied_dynamic_move_count": len(dynamic_moves),
        "applied_time_domain_action_count": len([r for r in time_reports if isinstance(r, dict) and r.get("active")]),
        "applied_stereo_side_action_count": len([r for r in stereo_side_reports if isinstance(r, dict) and r.get("active")]),
        "time_domain_reports": time_reports,
        "stereo_side_reports": stereo_side_reports,
        "v59_2_program_dependent_dynamics_stereo_side": v59_2,
    })
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), report

def _apply_output_dither(y: np.ndarray, bit_depth: int = 24) -> tuple[np.ndarray, dict[str, Any]]:
    if not _env_bool("BUSY_DITHER", True):
        return y, {"active": False, "reason": "disabled"}
    bit_depth = int(np.clip(bit_depth, 16, 24))
    lsb = 1.0 / float(2 ** (bit_depth - 1))
    rng = np.random.default_rng(20260613)
    noise = (rng.random(y.shape) - rng.random(y.shape)) * lsb
    shaped = _env_bool("BUSY_NOISE_SHAPING", True)
    if shaped and y.shape[0] > 8:
        # Very light high-passed TPDF dither. This is intentionally tiny for 24-bit output.
        try:
            b = np.array([1.0, -0.85], dtype=np.float64)
            a = np.array([1.0], dtype=np.float64)
            noise = lfilter(b, a, noise, axis=0)
        except Exception:
            shaped = False
    out = np.asarray(y, dtype=np.float64) + noise
    out = np.clip(out, -1.0, 1.0)
    return out, {"active": True, "bit_depth": bit_depth, "noise_shaping": shaped, "tpdf_lsb": round(lsb, 12)}

def _render_once(
    y: np.ndarray,
    sr: int,
    decision: dict[str, Any],
    mode: str,
    character_scale: float = 1.0,
    loudness_offset_db: float = 0.0,
    mb_amount_scale: float = 1.0,
    final_loudness_stage: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    d = _normalize_decision(decision)
    v59_2_plan = (d.get("v59_lf_crest_source_conditioning_plan") or {}).get("v59_2_program_dependent_dynamics_stereo_side", {}) if isinstance(d.get("v59_lf_crest_source_conditioning_plan"), dict) else {}
    v59_2_controls = v59_2_plan.get("program_dynamics_controls", {}) if isinstance(v59_2_plan, dict) and isinstance(v59_2_plan.get("program_dynamics_controls"), dict) else {}
    v59_2_mb_scale = float(np.clip(float(v59_2_controls.get("multiband_amount_scale", 1.0) or 1.0), 0.45, 1.10))
    v59_2_dyn_scale = float(np.clip(float(v59_2_controls.get("dynamic_eq_intensity_scale", 1.0) or 1.0), 0.55, 1.08))
    v59_2_upward_scale = float(np.clip(float(v59_2_controls.get("upward_density_scale", 1.0) or 1.0), 0.20, 1.10))
    defaults = _mode_defaults(mode)
    chain = _genre_chain_params(d, mode)
    report: dict[str, Any] = {
        "active": False,
        "actions": [],
        "stages": [],
        "genre_chain": chain,
        "spatial_fx_mastering": d.get("spatial_fx_mastering_plan", {}),
        "final_loudness_stage": bool(final_loudness_stage),
        "v6316_true_pre_limiter_render": {
            "schema_version": "busy_v6316_true_pre_limiter_render",
            "active": bool(not final_loudness_stage),
            "policy": "When active, _render_once returns the EQ/dynamics/glue/polish checkpoint before loudness_iteration/limiter so downstream commercial finish, v60 DML, final limiter, dither, export lock, and QC run in their intended order.",
        },
        "v59_2_program_dependent_dynamics_controls": {
            "active": bool(v59_2_plan.get("active") or any(abs(x - 1.0) > 1e-4 for x in (v59_2_mb_scale, v59_2_dyn_scale, v59_2_upward_scale))),
            "multiband_amount_scale": round(v59_2_mb_scale, 4),
            "dynamic_eq_intensity_scale": round(v59_2_dyn_scale, 4),
            "upward_density_scale": round(v59_2_upward_scale, 4),
        },
    }

    out = highpass(y, sr, cutoff_hz=24.0, order=2)
    report["stages"].append("highpass_24hz")

    # Conditional Suno Repair Stage. This is bypassed when artifact analysis says
    # the source is already clean; not every Suno WAV needs repair.
    repair = d.get("suno_repair", {}) if isinstance(d.get("suno_repair", {}), dict) else {}
    repair_active = bool(repair.get("active", False))
    repair_intensity = float(repair.get("intensity", 0.0) or 0.0)
    if repair_active and repair_intensity > 0.02:
        repair_static = _collect_moves(d, "suno_repair_static_eq")
        repair_dyn = _collect_moves(d, "suno_repair_dynamic_eq")
        if repair_static:
            out = apply_eq_moves(out, sr, repair_static, scale=min(1.0, repair_intensity))
            report["stages"].append(f"suno_repair_static_{repair_intensity:.2f}")
        if repair_dyn:
            out = dynamic_eq(out, sr, repair_dyn, intensity=min(1.25, 0.65 + repair_intensity))
            report["stages"].append(f"suno_repair_dynamic_{repair_intensity:.2f}")
        report["suno_repair"] = repair
    else:
        report["suno_repair"] = {"active": False, "reason": repair.get("reason", "not requested")}

    out, mix_conditioning_report = apply_mix_conditioning_audio(out, sr, d.get("mix_conditioning_plan", {}))
    report["mix_conditioning"] = mix_conditioning_report
    if mix_conditioning_report.get("active"):
        report["stages"].append("mix_conditioning")

    out = apply_eq_moves(out, sr, _collect_moves(d, "corrective_eq"), scale=1.0)
    report["stages"].append("corrective_eq")

    effective_character_scale = float(character_scale) * float(chain.get("character_eq_multiplier", 1.0))
    effective_character_scale = float(np.clip(effective_character_scale, 0.35, 3.10))
    out = apply_eq_moves(out, sr, _collect_moves(d, "character_eq"), scale=effective_character_scale)
    report["stages"].append(f"character_eq_scale_{effective_character_scale:.2f}")

    ms = d.get("ms", {})
    out = ms_low_mono(out, sr, cutoff_hz=float(ms.get("mono_below_hz", 120)), amount=float(ms.get("low_side_reduction", 0.85)))
    out = adjust_width(out, width=float(ms.get("width", 1.0)))
    report["stages"].append("ms_low_mono_width")

    out, v58_spectral_report = _apply_v58_spectral_conditioning(out, sr, d)
    report["v58_spectral_conditioning"] = v58_spectral_report
    if v58_spectral_report.get("active"):
        report["stages"].append("v58_spectral_conditioning")

    out, v59_lf_crest_report = _apply_v59_lf_crest_source_conditioning(out, sr, d)
    report["v59_lf_crest_source_conditioning"] = v59_lf_crest_report
    if v59_lf_crest_report.get("render_affecting"):
        report["stages"].append("v59_lf_crest_source_conditioning")

    dyn_intensity = float(d.get("dynamic_eq_intensity", 1.0)) * min(1.45, max(0.70, character_scale)) * float(chain.get("dynamic_eq_multiplier", 1.0)) * v59_2_dyn_scale
    dyn_intensity = float(np.clip(dyn_intensity, 0.35, 1.95))
    out = dynamic_eq(out, sr, _collect_moves(d, "dynamic_eq"), intensity=dyn_intensity)
    report["stages"].append(f"dynamic_eq_intensity_{dyn_intensity:.2f}")

    out, vocal_report = _vocal_lead_protection(out, sr, d)
    report["vocal_lead_protection"] = vocal_report
    if vocal_report.get("active"):
        report["stages"].append("vocal_lead_protection")

    mb = d.get("multiband", {}) or {}
    mb_cfg = d.get("multiband_compression", {}) or mb.get("multiband_compression", {}) or {}
    band_settings = mb_cfg.get("bands", []) if isinstance(mb_cfg, dict) else []
    mb_amount = float(mb.get("amount", defaults["mb_amount"])) * mb_amount_scale * float(chain.get("multiband_multiplier", 1.0)) * v59_2_mb_scale
    mb_amount = float(np.clip(mb_amount, 0.18, 1.85))
    if band_settings:
        out = multiband_compress_custom(out, sr, band_settings, amount=mb_amount)
        report["stages"].append("custom_multiband")
    else:
        out = multiband_compress(out, sr, mode=mb.get("mode", defaults["mb_mode"]), amount=mb_amount)
        report["stages"].append("fallback_multiband")

    flags = set(d.get("processing_flags", []) or [])
    upward_mix = float(chain.get("upward_density_mix", 0.0) or 0.0)
    if "allow_upward_compression_light" in flags or "allow_parallel_density_light" in flags:
        upward_mix = max(upward_mix, 0.055 if "Clean" not in mode else 0.035)
    upward_mix *= v59_2_upward_scale
    if upward_mix > 0:
        out = upward_parallel_density(out, sr, threshold_db=-40.0, ratio=1.45, mix=upward_mix)
        report["stages"].append(f"upward_density_mix_{upward_mix:.3f}")

    out, loudprep_report = _apply_commercial_loudness_prep(out, sr, d)
    report["commercial_loudness_prep"] = loudprep_report
    if loudprep_report.get("active"):
        report["stages"].append("commercial_loudness_prep")

    out, imager_report = _intelligent_multiband_imager(out, sr, d)
    report["intelligent_multiband_imager"] = imager_report
    if imager_report.get("active"):
        report["stages"].append("intelligent_multiband_imager")

    out, transient_report = _transient_punch_guard(out, sr, d)
    report["transient_punch_guard"] = transient_report
    if transient_report.get("active"):
        report["stages"].append("transient_punch_guard")

    apply_sat, sat_drive, sat_mix, clip_drive, clip_mix = _saturation_params(d, mode, chain=chain)
    if apply_sat and sat_drive > 0 and sat_mix > 0:
        out = subtle_saturation(out, drive_db=sat_drive, mix=sat_mix)
        report["stages"].append(f"saturation_drive_{sat_drive:.2f}_mix_{sat_mix:.2f}")

    out = soft_clip(out, drive_db=clip_drive, mix=clip_mix)
    if clip_drive > 0 and clip_mix > 0:
        report["stages"].append(f"soft_clip_drive_{clip_drive:.2f}_mix_{clip_mix:.2f}")

    target_lufs, ceiling_db = _target_from_decision(d, mode)
    loudprep = d.get("commercial_loudness_prep", {}) if isinstance(d.get("commercial_loudness_prep", {}), dict) else {}
    loudprep_extra_push = float(loudprep.get("extra_push_db", 0.0) or 0.0) if loudprep.get("active") else 0.0
    target_lufs += loudness_offset_db + float(chain.get("loudness_offset_db", 0.0) or 0.0) + loudprep_extra_push
    v6314_2_target_guard = _v6314_2_bamix_loudness_target_guard(d, target_lufs, mode)
    if v6314_2_target_guard.get("active") and v6314_2_target_guard.get("applied"):
        try:
            target_lufs = float(v6314_2_target_guard.get("effective_target_lufs", target_lufs))
        except Exception:
            pass
    report["v6314_2_bamix_loudness_target_guard"] = v6314_2_target_guard
    if not bool(final_loudness_stage):
        # v63.1.6: true pre-limiter handoff.  Do not apply the loudness
        # iteration/limiter here; finalize_limiter_master owns the finalizer
        # stages and must receive an un-limited post-processing checkpoint.
        report["active"] = True
        report["loudness_iteration"] = {
            "schema_version": "busy_loudness_iteration_v6316_true_pre_limiter_skip",
            "enabled": True,
            "skipped": True,
            "reason": "v6316_true_pre_limiter_handoff_before_loudness_iteration",
            "target_lufs_preview": round(float(target_lufs), 3),
            "ceiling_db_preview": round(float(ceiling_db), 3),
            "policy": "Limiter/loudness iteration is deliberately skipped in the pre-limiter render so existing mastering/finalizer stages are not short-circuited by an already-limited checkpoint.",
        }
        report["v6316_true_pre_limiter_render"].update({
            "active": True,
            "returned_before_loudness_iteration": True,
            "target_lufs_preview": round(float(target_lufs), 3),
            "ceiling_db_preview": round(float(ceiling_db), 3),
        })
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False), report
    out, iter_report = _loudness_iteration(out, sr, target_lufs=target_lufs, ceiling_db=ceiling_db, limiter=d.get("limiter", {}), max_iters=7)
    iter_report["commercial_loudness_prep_extra_push_db"] = round(loudprep_extra_push, 3)
    iter_report["v6314_2_bamix_loudness_target_guard"] = v6314_2_target_guard
    report["loudness_iteration"] = iter_report
    # v8.5.3.15: temporal drift is a pre-clean/preflight concern, not a repeated
    # post-render process.  Re-running it after every limiter pass can over-correct
    # musical section changes and makes the pipeline order ambiguous.  Keep an
    # explicit opt-in escape hatch for experiments only.
    if _env_bool("BUSY_TEMPORAL_DRIFT_IN_RENDER", False):
        try:
            out, drift_report = stabilize_temporal_spectral_drift(out, sr)
            report["temporal_spectral_drift"] = drift_report
            if isinstance(drift_report, dict) and drift_report.get("active"):
                report["active"] = True
                report.setdefault("actions", []).append("temporal_spectral_drift_stabilized")
        except Exception as e:
            report["temporal_spectral_drift"] = {
                "schema_version": "busy_temporal_spectral_drift_v1",
                "enabled": True,
                "active": False,
                "reason": "error",
                "actions": [],
                "selected_corrections_count": 0,
                "error_type": type(e).__name__,
                "error": str(e)[:300],
            }
    else:
        report["temporal_spectral_drift"] = {
            "schema_version": "busy_temporal_spectral_drift_v1",
            "enabled": True,
            "active": False,
            "reason": "handled_by_pre_clean_or_preflight",
            "actions": [],
            "selected_corrections_count": 0,
            "policy": "render-stage temporal drift correction disabled by default in v8.5.3.15",
        }

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return out, report


def master_audio(y: np.ndarray, sr: int, decision: dict[str, Any], mode: str = "Auto Commercial Master", pre_analysis: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Run report-driven mastering with post-analysis rollback.

    The chain reflects the v6 genre-chain reports: working-stage normalization, CEQ/ChEQ separation,
    M/S low-end control, dynamic EQ, genre-family multiband, saturation/upward density,
    soft clip, lookahead limiting, loudness iteration, and quality-gate rollback.
    """
    runtime_intensity_lock = _refresh_runtime_mastering_policy(decision)
    _debug("engine_start", sr=sr, shape=getattr(y, "shape", None), dtype=str(getattr(y, "dtype", "")), mode=mode, runtime_intensity_lock=runtime_intensity_lock)
    if pre_analysis is not None:
        before = pre_analysis
        _debug("engine_done_input_analysis", reused_precomputed=True, lufs=before.get("integrated_lufs"), true_peak=before.get("approx_true_peak_dbfs"), crest=before.get("crest_factor_db"), correlation=before.get("correlation"))
    else:
        before = analyze_audio(y, sr)
        _debug("engine_done_input_analysis", reused_precomputed=False, lufs=before.get("integrated_lufs"), true_peak=before.get("approx_true_peak_dbfs"), crest=before.get("crest_factor_db"), correlation=before.get("correlation"))

    original_before = before
    y, residue_pre_clean_stage, before = _apply_residue_pre_clean_once(y, sr, before, decision, mode)
    _debug(
        "engine_done_residue_pre_clean_stage",
        active=residue_pre_clean_stage.get("active"),
        bypassed=residue_pre_clean_stage.get("bypassed"),
        reason=residue_pre_clean_stage.get("reason"),
        action_count=len(residue_pre_clean_stage.get("actions") or []),
    )

    governed_decision, governor_report = _loudness_push_governor(before, decision, mode)
    decision = governed_decision
    _debug("engine_done_loudness_push_governor", **governor_report)
    decision, commercial_loudness_prep = _attach_commercial_loudness_prep(before, decision, mode)
    _debug("engine_done_commercial_loudness_prep_plan", **{k: v for k, v in commercial_loudness_prep.items() if k not in {"static_moves", "dynamic_moves"}})
    decision, v62_9_9_research_governance = _attach_v6299_research_governance(before, decision, mode)
    _debug(
        "engine_done_v62_9_9_research_governance",
        active=v62_9_9_research_governance.get("active"),
        micro=(v62_9_9_research_governance.get("microdynamics_lra_shape_recovery") or {}).get("active") if isinstance(v62_9_9_research_governance.get("microdynamics_lra_shape_recovery"), dict) else None,
        section=(v62_9_9_research_governance.get("section_aware_spectral_consistency") or {}).get("active") if isinstance(v62_9_9_research_governance.get("section_aware_spectral_consistency"), dict) else None,
        texture=(v62_9_9_research_governance.get("musical_texture_preservation_gate") or {}).get("active") if isinstance(v62_9_9_research_governance.get("musical_texture_preservation_gate"), dict) else None,
        low_end=(v62_9_9_research_governance.get("translation_aware_low_end_governor") or {}).get("active") if isinstance(v62_9_9_research_governance.get("translation_aware_low_end_governor"), dict) else None,
    )
    decision, v58_spectral_conditioning_plan = _attach_v58_spectral_conditioning(before, decision, mode)
    _debug("engine_done_v58_spectral_conditioning_plan", active=v58_spectral_conditioning_plan.get("active"), reasons=v58_spectral_conditioning_plan.get("reasons"), static_moves=v58_spectral_conditioning_plan.get("static_move_count"), dynamic_moves=v58_spectral_conditioning_plan.get("dynamic_move_count"))
    decision, v59_lf_crest_source_conditioning_plan = _attach_v59_lf_crest_source_conditioning(before, decision, mode)
    _debug("engine_done_v59_lf_crest_source_conditioning_plan", active=v59_lf_crest_source_conditioning_plan.get("active"), render_affecting=v59_lf_crest_source_conditioning_plan.get("render_affecting"), reasons=v59_lf_crest_source_conditioning_plan.get("reasons"), dynamic_moves=v59_lf_crest_source_conditioning_plan.get("dynamic_move_count"), time_actions=v59_lf_crest_source_conditioning_plan.get("time_domain_action_count"), stereo_side_actions=v59_lf_crest_source_conditioning_plan.get("stereo_side_action_count"))
    decision, v62_8_mastering_blocker_preparation = _attach_v62_8_mastering_blocker_preparation(before, decision, mode)
    _debug("engine_done_v62_8_mastering_blocker_preparation", active=v62_8_mastering_blocker_preparation.get("active"), actions=v62_8_mastering_blocker_preparation.get("action_count"), families=v62_8_mastering_blocker_preparation.get("micro_ab_candidate_families"), decision=(v62_8_mastering_blocker_preparation.get("commercial_loudness_feasibility") or {}).get("decision"))

    staged, stage_gain_db = _stage_to_working_lufs(y, sr, WORKING_STAGE_LUFS)
    character_scale = _boldness_scale(decision, mode)
    _debug("engine_done_working_stage", stage_gain_db=round(stage_gain_db, 3), character_scale=round(character_scale, 3), staged_shape=getattr(staged, "shape", None))

    # v63.1.6.2.1: the default single-stage Cloud Run path still used the
    # legacy limited first pass directly, so the reconnect logic added to the
    # A-2b prepared-stage path was not actually visible in normal runs.  Mirror
    # the same true pre-limiter checkpoint and v61 measured-feedback call here.
    v6316_true_pre_limiter_handoff: dict[str, Any] = {"enabled": bool(_env_bool("BUSY_V6316_TRUE_PRE_LIMITER_HANDOFF", True)), "active": False, "reason": "disabled"}
    if bool(v6316_true_pre_limiter_handoff.get("enabled")):
        try:
            _debug("engine_start_v631621_true_pre_limiter_handoff_render")
            _handoff_audio, _handoff_proc = _render_once(
                staged,
                sr,
                decision,
                mode,
                character_scale=character_scale,
                loudness_offset_db=0.0,
                mb_amount_scale=1.0,
                final_loudness_stage=False,
            )
            _handoff_analysis = analyze_audio_fast_qc(_handoff_audio, sr, true_peak_oversample=_qc_oversample_factor())
            v6316_true_pre_limiter_handoff = {
                "schema_version": "busy_v631621_true_pre_limiter_handoff_single_stage",
                "enabled": True,
                "active": True,
                "returned_checkpoint": "true_pre_limiter_before_loudness_iteration",
                "analysis": _analysis_summary_for_loudness(_handoff_analysis),
                "stages": (_handoff_proc or {}).get("stages", []),
                "policy": "Single-stage master_audio path now exposes the same true pre-limiter checkpoint as the prepared-stage path; it is telemetry/handoff authority, not the delivered final buffer.",
            }
            _debug("engine_done_v631621_true_pre_limiter_handoff_render", lufs=_handoff_analysis.get("integrated_lufs"), true_peak=_handoff_analysis.get("approx_true_peak_dbfs"), stages=(_handoff_proc or {}).get("stages", []))
        except Exception as exc:
            v6316_true_pre_limiter_handoff = {"schema_version": "busy_v631621_true_pre_limiter_handoff_single_stage", "enabled": True, "active": False, "reason": "exception", "error": str(exc)[:240]}
            _debug("engine_error_v631621_true_pre_limiter_handoff_render", error=str(exc)[:240])

    _debug("engine_start_render_first_pass")
    first, first_proc = _render_once(staged, sr, decision, mode, character_scale=character_scale, loudness_offset_db=0.0, mb_amount_scale=1.0)
    _debug("engine_done_render_first_pass", stages=first_proc.get("stages", []), loudness_iteration=first_proc.get("loudness_iteration"))
    first_analysis = analyze_audio_fast_qc(first, sr, true_peak_oversample=_qc_oversample_factor())
    reasons = _safety_risk(before, first_analysis, decision)
    _debug("engine_done_first_pass_qc", reasons=reasons, lufs=first_analysis.get("integrated_lufs"), true_peak=first_analysis.get("approx_true_peak_dbfs"), crest=first_analysis.get("crest_factor_db"), correlation=first_analysis.get("correlation"))

    final = first
    final_analysis = first_analysis
    final_proc = first_proc
    rollback_used = False
    rollback_attempts: list[dict[str, Any]] = []

    v61_feedback_optimizer: dict[str, Any] = {"enabled": False, "active": False, "attempted": False, "reason": "not_run"}
    if _env_bool("BUSY_V61_RENDER_MEASURED_FEEDBACK_OPTIMIZER", True):
        _v631621_prepare_proxy = {
            "input_analysis": before,
            "loudness_push_governor": governor_report,
            "commercial_loudness_prep": commercial_loudness_prep,
            "working_stage_gain_db": round(stage_gain_db, 3),
            "character_scale_initial": round(character_scale, 3),
            "v6316_true_pre_limiter_handoff": v6316_true_pre_limiter_handoff,
        }
        final, final_analysis, final_proc, reasons, v61_feedback_optimizer = _v61_render_measured_feedback_optimizer(
            staged,
            sr,
            decision,
            _v631621_prepare_proxy,
            mode,
            character_scale,
            first,
            first_analysis,
            first_proc,
        )
        rollback_used = bool(v61_feedback_optimizer.get("rollback_count", 0))
        if v61_feedback_optimizer.get("active"):
            rollback_attempts.append({
                "type": "v61_render_measured_feedback_optimizer",
                "selected_pass": v61_feedback_optimizer.get("selected_pass"),
                "accepted_pass_count": v61_feedback_optimizer.get("accepted_pass_count"),
                "rollback_count": v61_feedback_optimizer.get("rollback_count"),
                "selected_score": v61_feedback_optimizer.get("selected_score"),
            })
        _debug(
            "engine_done_v61_feedback_optimizer",
            active=v61_feedback_optimizer.get("active"),
            attempted=v61_feedback_optimizer.get("attempted"),
            selected_pass=v61_feedback_optimizer.get("selected_pass"),
            accepted_pass_count=v61_feedback_optimizer.get("accepted_pass_count"),
            rollback_count=v61_feedback_optimizer.get("rollback_count"),
            selected_score=v61_feedback_optimizer.get("selected_score"),
            stop_reason=v61_feedback_optimizer.get("stop_reason"),
        )

    # Multi-step rollback: reduce character/loudness first, then restore a hair of transient if over-crushed.
    if reasons and not bool(v61_feedback_optimizer.get("selected_pass")):
        rollback_used = True
        _debug("engine_start_rollback", initial_reasons=reasons)
        all_candidates = [
            {"character_scale": character_scale * 0.78, "loudness_offset_db": -0.35, "mb_amount_scale": 0.92},
            {"character_scale": character_scale * 0.62, "loudness_offset_db": -0.75, "mb_amount_scale": 0.80},
            {"character_scale": character_scale * 0.48, "loudness_offset_db": -1.10, "mb_amount_scale": 0.68},
        ]
        max_attempts = _effective_rollback_max_attempts(staged, sr, decision)
        # Deep rollback is expensive because it re-renders the whole pre-limiter chain.
        # On long role-first tracks, auto-cap to avoid Streamlit resource kills.
        candidates = all_candidates[:max_attempts]
        _debug("engine_rollback_plan", rollback_max_attempts=max_attempts, candidate_count=len(candidates))
        best = (len(reasons), final, final_analysis, final_proc, reasons)
        for idx, c in enumerate(candidates, start=1):
            _debug("engine_start_rollback_attempt", attempt=idx, settings=c)
            cand, proc = _render_once(staged, sr, decision, mode, **c)
            cand_analysis = analyze_audio_fast_qc(cand, sr, true_peak_oversample=_qc_oversample_factor())
            cand_reasons = _safety_risk(before, cand_analysis, decision)
            rollback_attempts.append({"settings": c, "warnings": cand_reasons})
            _debug("engine_done_rollback_attempt", attempt=idx, warnings=cand_reasons, lufs=cand_analysis.get("integrated_lufs"), true_peak=cand_analysis.get("approx_true_peak_dbfs"), crest=cand_analysis.get("crest_factor_db"))
            score = len(cand_reasons)
            if score < best[0] or (score == best[0] and cand_analysis.get("crest_factor_db", 0) > best[2].get("crest_factor_db", 0)):
                best = (score, cand, cand_analysis, proc, cand_reasons)
        _, final, final_analysis, final_proc, reasons = best
        _debug("engine_done_rollback", remaining_reasons=reasons, selected_lufs=final_analysis.get("integrated_lufs"), selected_true_peak=final_analysis.get("approx_true_peak_dbfs"), selected_crest=final_analysis.get("crest_factor_db"))

    # If limiter still crushed the track but otherwise safe, add micro transient restoration and re-limit.
    targets = decision.get("targets", {}) if isinstance(decision, dict) else {}
    crest_floor = float(targets.get("crest_floor_db", 5.0))
    if final_analysis.get("crest_factor_db", 99) < crest_floor and "true_peak_too_hot" not in reasons:
        _debug("engine_start_micro_transient_restore", crest=final_analysis.get("crest_factor_db"), crest_floor=crest_floor)
        restored = transient_micro_expander(final, sr, amount=0.06)
        restored = lookahead_limiter(restored, sr, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=1.0, release_ms=140)
        restored_analysis = analyze_audio_fast_qc(restored, sr, true_peak_oversample=_qc_oversample_factor())
        restored_reasons = _safety_risk(before, restored_analysis, decision)
        if len(restored_reasons) <= len(reasons) and restored_analysis.get("crest_factor_db", 0) >= final_analysis.get("crest_factor_db", 0):
            final, final_analysis, reasons = restored, restored_analysis, restored_reasons
            final_proc.setdefault("stages", []).append("micro_transient_restore")
            _debug("engine_done_micro_transient_restore", reasons=reasons, crest=final_analysis.get("crest_factor_db"), true_peak=final_analysis.get("approx_true_peak_dbfs"))

    # Memory-safe final oversampled limiter. Runs in chunks to avoid full-song 8x/16x/32x allocation.
    os_factor, adaptive_quality_report = _effective_oversample_factor(before, decision)
    limiter_cfg = decision.get("limiter", {}) if isinstance(decision, dict) else {}
    lookahead_ms = limiter_cfg.get("lookahead_ms", 1.0)
    release_ms = limiter_cfg.get("release_ms", 160.0)
    if isinstance(lookahead_ms, list):
        lookahead_ms = float(sum(lookahead_ms) / len(lookahead_ms))
    if isinstance(release_ms, list):
        release_ms = float(sum(release_ms) / len(release_ms))
    _debug("engine_start_oversampled_final_limiter", oversample=os_factor, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=float(lookahead_ms), release_ms=float(release_ms))
    final = oversampled_limit_chunked(
        final,
        sr,
        ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
        oversample=os_factor,
        lookahead_ms=float(lookahead_ms),
        release_ms=float(release_ms),
    )
    _debug("engine_done_oversampled_final_limiter", output_shape=getattr(final, "shape", None))
    _debug("engine_start_true_peak_ceiling_attenuation", oversample=os_factor, target_db=FINAL_TRUE_PEAK_CEILING_DB)
    final, final_true_peak_gain_db, true_peak_ceiling_attenuation = _attenuate_to_true_peak_ceiling(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=os_factor)
    _debug("engine_done_true_peak_ceiling_attenuation", gain_db=round(final_true_peak_gain_db, 3), active=true_peak_ceiling_attenuation.get("active"), reason=true_peak_ceiling_attenuation.get("reason"), pre_true_peak_dbtp=true_peak_ceiling_attenuation.get("pre_true_peak_dbtp"))

    final, optimizer_report = _multi_pass_auto_optimizer(final, sr, before, decision, mode)
    if optimizer_report.get("active"):
        _debug("engine_done_multi_pass_optimizer", selected=optimizer_report.get("selected"), selected_score=optimizer_report.get("selected_score"), base_score=optimizer_report.get("base_score"))

    experimental_residue_light_control = residue_pre_clean_stage
    _debug(
        "engine_skip_post_residue_light_control",
        reason="residue_cleanup_is_pre_master_once",
        pre_clean_active=experimental_residue_light_control.get("active"),
        action_count=len(experimental_residue_light_control.get("actions") or []),
    )

    # v8.5.3.19: commercial finish must evaluate and apply the virtual strategy
    # against the pre-limiter signal, not against an already limited master.
    # Running the hot solver after the first safety limiter made +5~+11 dB
    # candidates collapse back into the limiter and report almost no LUFS gain.
    # v63.1.9: single-stage true pre-limiter handoff is not just telemetry.
    # If the checkpoint render succeeded, commercial finish must use it rather
    # than the already-limited standard final.
    if "_handoff_audio" in locals() and isinstance(v6316_true_pre_limiter_handoff, dict) and v6316_true_pre_limiter_handoff.get("active"):
        commercial_source_for_finish = _handoff_audio
        commercial_source_label = "true_pre_limiter_handoff"
    elif "pre_limiter" in locals():
        commercial_source_for_finish = pre_limiter
        commercial_source_label = "pre_limiter"
    else:
        commercial_source_for_finish = final
        commercial_source_label = "standard_final"
    _commercial_finish_started_at = time.monotonic()
    _debug(
        "engine_start_commercial_loudness_finish",
        source=commercial_source_label,
        shape=getattr(commercial_source_for_finish, "shape", None),
        sr=sr,
    )
    commercial_candidate, commercial_finish_report = _commercial_loudness_finish_pass(commercial_source_for_finish, sr, before, decision, mode)
    try:
        commercial_finish_report["single_stage_commercial_finish_source"] = commercial_source_label
        commercial_finish_report["single_stage_true_pre_limiter_handoff_connected"] = bool(commercial_source_label in {"true_pre_limiter_handoff", "pre_limiter"})
        commercial_finish_report["v6319_bamix_handoff_mode"] = _bamix_v6319_handoff_mode(decision)
    except Exception:
        pass
    try:
        commercial_finish_report["runtime_elapsed_sec"] = round(max(0.0, time.monotonic() - _commercial_finish_started_at), 3)
    except Exception:
        pass
    _debug(
        "engine_done_commercial_loudness_finish_pass",
        selected=commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None,
        reason=commercial_finish_report.get("reason") if isinstance(commercial_finish_report, dict) else None,
        elapsed_sec=commercial_finish_report.get("runtime_elapsed_sec") if isinstance(commercial_finish_report, dict) else None,
        virtual_runtime=(commercial_finish_report.get("virtual_hot_commercial_solver") or {}).get("runtime_proxy_guard") if isinstance(commercial_finish_report, dict) and isinstance(commercial_finish_report.get("virtual_hot_commercial_solver"), dict) else None,
    )
    commercial_selected = bool(commercial_finish_report.get("selected") and commercial_finish_report.get("selected") != "base")
    commercial_finish_report["strategy_application_stage"] = "pre_limiter_before_final_safety_limiter"
    # v8.5.3.20: never replace the already rendered standard final with a
    # commercial candidate that is quieter.  The finish pass is evaluated against
    # the pre-limiter source, so its internal base can be much quieter than the
    # standard final.  This guard prevents a "selected" pre-limiter candidate from
    # making the delivered master smaller.
    standard_final_audio_for_fallback = final
    standard_final_reference = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    commercial_finish_report["standard_final_reference_analysis"] = _analysis_summary_for_loudness(standard_final_reference)
    commercial_finish_report["v6316_standard_final_loudness_iteration"] = standard_final_loudness_iteration if 'standard_final_loudness_iteration' in locals() else {}
    if commercial_selected:
        _selected_fg_limiter_cfg = {**(limiter_cfg if isinstance(limiter_cfg, dict) else {}), "candidate_analysis": commercial_finish_report.get("output") if isinstance(commercial_finish_report, dict) else None, "candidate_label": commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None}
        commercial_candidate, _final_grade_report = _final_grade_selected_candidate_render(
            commercial_candidate,
            sr,
            before,
            decision,
            limiter_cfg=_selected_fg_limiter_cfg,
        )
        commercial_finish_report["selected_candidate_final_grade_render"] = _final_grade_report
        if isinstance(_final_grade_report, dict):
            _v56_pref_output = _final_grade_report.get("post_analysis_full") if isinstance(_final_grade_report.get("post_analysis_full"), dict) and _final_grade_report.get("post_analysis_full") else _final_grade_report.get("post_analysis")
            if isinstance(_v56_pref_output, dict):
                commercial_finish_report["output"] = _v56_pref_output
                commercial_finish_report["output_analysis_mode"] = "full_analyze_audio" if _final_grade_report.get("post_analysis_full") else "fast_qc"
            if isinstance(_final_grade_report.get("buffer_identity"), dict):
                commercial_finish_report["selected_candidate_export_identity"] = _final_grade_report.get("buffer_identity")
            commercial_finish_report["v56_candidate_handoff_export_identity_lock"] = {
                "schema_version": "busy_v56_candidate_handoff_export_identity_lock",
                "active": True,
                "selected_candidate": commercial_finish_report.get("selected"),
                "expected_analysis_mode": commercial_finish_report.get("output_analysis_mode"),
                "selected_candidate_buffer_sha256_16": (_final_grade_report.get("buffer_identity") or {}).get("sha256_16") if isinstance(_final_grade_report.get("buffer_identity"), dict) else None,
                "policy": "selected-candidate output uses full export analysis when available so export lock compares the same analyzer family.",
            }
            # v8.5.3.57.1: the v57 damage gate must be applied to the
            # final-grade selected-candidate render as well, not only to the
            # earlier candidate checkpoint. v56 can legitimately make the
            # final-grade buffer louder than the checkpoint, so LRA/transient
            # damage must be checked on the actual buffer that would replace the
            # standard final.
            try:
                _v57_final_grade_analysis = _v56_pref_output if isinstance(_v56_pref_output, dict) else {}
                _v57_final_grade_warnings = _safety_risk(before, _v57_final_grade_analysis, decision) if _v57_final_grade_analysis else []
                _v57_final_grade_profile = _commercial_loudness_finish_profile(decision, mode)
                _v57_final_grade_gate = _v57_damage_aware_candidate_gate(
                    before,
                    _v57_final_grade_analysis,
                    standard_final_reference,
                    decision,
                    _v57_final_grade_profile,
                    _v57_final_grade_warnings,
                    float(commercial_finish_report.get("commercial_floor_lufs", commercial_finish_report.get("target_lufs", -9.0)) or -9.0),
                    float(commercial_finish_report.get("target_lufs", -8.0) or -8.0),
                ) if _v57_final_grade_analysis else {"enabled": _env_bool("BUSY_V57_DAMAGE_AWARE_SAFE_SELECTION", True), "active": False, "reason": "missing_final_grade_analysis"}
                commercial_finish_report["selected_candidate_final_grade_v57_warnings"] = _v57_final_grade_warnings
                commercial_finish_report["selected_candidate_final_grade_v57_damage_gate"] = _v57_final_grade_gate
                if isinstance(_final_grade_report, dict):
                    _final_grade_report["v57_damage_aware_candidate_gate"] = _v57_final_grade_gate
                    _final_grade_report["v57_warnings"] = _v57_final_grade_warnings
                if _v57_final_grade_gate.get("reject"):
                    commercial_selected = False
                    commercial_finish_report["candidate_rejected_after_v57_final_grade_damage_gate"] = True
                    commercial_finish_report["candidate_rejection_reason"] = "v57_final_grade_damage_gate"
                    commercial_finish_report["candidate_rejection_problem_set"] = _v57_final_grade_gate.get("blockers", [])
                    commercial_finish_report["fallback_final_source"] = "standard_final_after_v57_damage_gate"
                    commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
                    commercial_finish_report["output"] = commercial_finish_report.get("standard_final_reference_analysis") or commercial_finish_report.get("output")
                    commercial_finish_report["output_analysis_mode"] = "standard_final_after_v57_damage_gate"
                _v57_attach_damage_aware_selection_summary(commercial_finish_report)
            except Exception as exc:
                commercial_finish_report["selected_candidate_final_grade_v57_damage_gate"] = {
                    "schema_version": "busy_v57_final_grade_damage_gate_v8_5_3_57_2",
                    "active": False,
                    "reason": "exception",
                    "error": str(exc)[:240],
                }
    # v63.1.6.2: a safe checkpoint below the commercial floor is not a final
    # commercial master.  Try one bounded post-safety floor recovery candidate;
    # if it fails, mark the checkpoint as partial and fall back to the standard
    # final instead of carrying stale selected/output telemetry downstream.
    if commercial_selected and _env_bool("BUSY_V63162_COMMERCIAL_FLOOR_RECOVERY", True):
        try:
            _floor = float(commercial_finish_report.get("commercial_floor_lufs", commercial_finish_report.get("target_lufs", -99.0)) or -99.0)
            _cand_out = commercial_finish_report.get("output") if isinstance(commercial_finish_report.get("output"), dict) else {}
            _cand_lufs_now = float(_cand_out.get("integrated_lufs", -99.0) or -99.0)
            _floor_tol = _env_float("BUSY_V63162_COMMERCIAL_FLOOR_TOLERANCE_LU", 0.04, 0.0, 0.20)
            if _floor > -90.0 and _cand_lufs_now < _floor - _floor_tol:
                _debug("commercial_finish_v63162_floor_recovery_start", selected=commercial_finish_report.get("selected"), candidate_lufs=round(float(_cand_lufs_now), 3), floor=round(float(_floor), 3))
                _recovered_audio, _recovered_analysis, _recovery_report = _v63162_commercial_floor_recovery_candidate(
                    commercial_candidate,
                    sr,
                    before,
                    decision,
                    mode,
                    standard_final_reference,
                    limiter_cfg=limiter_cfg if isinstance(limiter_cfg, dict) else {},
                    label=str(commercial_finish_report.get("selected") or "commercial_selected_candidate"),
                )
                commercial_finish_report["v63162_commercial_floor_recovery"] = _recovery_report
                if isinstance(_recovery_report, dict) and _recovery_report.get("accepted"):
                    commercial_candidate = _recovered_audio
                    commercial_finish_report["output"] = _analysis_summary_for_loudness(_recovered_analysis)
                    commercial_finish_report["output_analysis_mode"] = "v63162_post_safety_floor_recovery_fast_qc"
                    commercial_finish_report["target_met"] = True
                    commercial_finish_report["target_reached"] = True
                    commercial_finish_report["reason"] = "v63162_post_safety_floor_recovery_selected"
                    commercial_finish_report["selected"] = str(commercial_finish_report.get("selected") or "commercial_candidate") + "_floor_recovered"
                    _debug("commercial_finish_v63162_floor_recovery_accepted", lufs=_recovered_analysis.get("integrated_lufs"), true_peak=_recovered_analysis.get("approx_true_peak_dbfs"))
                else:
                    commercial_finish_report["partial_safe_checkpoint_not_final"] = True
                    commercial_finish_report["candidate_rejected_against_standard_final"] = True
                    commercial_finish_report["candidate_rejection_reason"] = "v63162_safe_checkpoint_below_floor_recovery_failed"
                    commercial_finish_report["candidate_vs_standard_lufs_delta_db"] = round(float(_cand_lufs_now - float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)), 3)
                    commercial_finish_report["fallback_final_source"] = "standard_final_limiter_output_after_v63162_floor_miss"
                    commercial_finish_report["selected_final_source"] = "standard_final_limiter_output"
                    commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
                    commercial_finish_report["output"] = _analysis_summary_for_loudness(standard_final_reference)
                    commercial_finish_report["delivered_final_output"] = _analysis_summary_for_loudness(standard_final_reference)
                    commercial_finish_report["target_met"] = bool(float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0) >= _floor - _floor_tol)
                    commercial_finish_report["reason"] = "v63162_safe_checkpoint_below_floor_standard_final_preserved"
                    commercial_selected = False
                    final = standard_final_audio_for_fallback
                    _debug("commercial_finish_v63162_floor_recovery_rejected", reason=(_recovery_report or {}).get("reason"), fallback="standard_final")
        except Exception as exc:
            commercial_finish_report["v63162_commercial_floor_recovery"] = {"active": False, "accepted": False, "reason": "exception", "error": str(exc)[:240]}

    if commercial_selected:
        _v6314_final_choice = _v6314_candidate_dynamic_governance(standard_final_reference, commercial_finish_report.get("output") if isinstance(commercial_finish_report.get("output"), dict) else {}, decision, float(commercial_finish_report.get("commercial_floor_lufs", -99.0) or -99.0), candidate_label=str(commercial_finish_report.get("selected") or "selected_candidate"))
        commercial_finish_report["v6314_cleaner_quieter_final_choice"] = _v6314_final_choice
        if _v6314_final_choice.get("reject_for_cleaner_quieter"):
            commercial_selected = False
            commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
            commercial_finish_report["candidate_rejected_after_v6314_cleaner_quieter_guard"] = True
            commercial_finish_report["candidate_rejection_reason"] = "v6314_cleaner_quieter_standard_final_preserved_after_target_met"
            commercial_finish_report["fallback_final_source"] = "standard_final_limiter_output"
            commercial_finish_report["selected"] = "standard_final_limiter_output"
            commercial_finish_report["selected_final_source"] = "standard_final_limiter_output"
            commercial_finish_report["delivered_final_is_standard_after_v6314_cleaner_guard"] = True
            commercial_finish_report["output"] = _analysis_summary_for_loudness(standard_final_reference)
            commercial_finish_report["delivered_final_output"] = _analysis_summary_for_loudness(standard_final_reference)
            try:
                _v6314_std_floor = float(commercial_finish_report.get("commercial_floor_lufs", -99.0) or -99.0)
                _v6314_std_lufs = float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)
                commercial_finish_report["target_met"] = bool(_v6314_std_lufs >= _v6314_std_floor)
                commercial_finish_report["target_reached"] = bool(_v6314_std_lufs >= float(commercial_finish_report.get("target_lufs", _v6314_std_floor) or _v6314_std_floor) - 0.20)
                commercial_finish_report["reason"] = "v6314_cleaner_quieter_standard_final_preserved_target_met" if commercial_finish_report["target_met"] else "v6314_cleaner_quieter_standard_final_preserved_below_floor"
            except Exception:
                commercial_finish_report["target_met"] = commercial_finish_report.get("target_met")
            final = standard_final_audio_for_fallback

    try:
        std_lufs = float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)
        cand_lufs = float((commercial_finish_report.get("output") or {}).get("integrated_lufs", -99.0) or -99.0)
    except Exception:
        std_lufs, cand_lufs = -99.0, -99.0
    if commercial_selected and cand_lufs < std_lufs + 0.12 and not bool(commercial_finish_report.get("target_met")):
        commercial_finish_report["candidate_rejected_against_standard_final"] = True
        _std_floor = float(commercial_finish_report.get("commercial_floor_lufs", -99.0) or -99.0)
        _std_target = float(commercial_finish_report.get("target_lufs", _std_floor) or _std_floor)
        _std_target_met = bool(std_lufs >= _std_target - 0.20)
        _std_floor_met = bool(std_lufs >= _std_floor - 0.03)
        commercial_finish_report["candidate_rejection_reason"] = "candidate_not_louder_than_standard_final_standard_preserved_target_met" if _std_floor_met else "candidate_not_louder_than_standard_final"
        commercial_finish_report["candidate_vs_standard_lufs_delta_db"] = round(float(cand_lufs - std_lufs), 3)
        commercial_finish_report["standard_final_target_met"] = _std_target_met
        commercial_finish_report["standard_final_floor_met"] = _std_floor_met
        if _std_floor_met:
            commercial_finish_report["target_met"] = True
            commercial_finish_report["target_reached"] = _std_target_met
            commercial_finish_report["reason"] = "standard_final_preserved_candidate_not_louder_target_met"
            commercial_finish_report["output"] = _analysis_summary_for_loudness(standard_final_reference)
            commercial_finish_report["delivered_final_output"] = _analysis_summary_for_loudness(standard_final_reference)
        commercial_selected = False
    # v8.5.3.21: before replacing the standard final, audit the candidate as
    # an actual final output.  A candidate that is louder but introduces a new
    # critical problem set is not accepted-with-warnings; it is rejected and the
    # standard final remains the delivered master.
    finish_profile_for_guard = _commercial_loudness_finish_profile(decision, mode)
    if commercial_selected:
        _debug("commercial_finish_final_replacement_guard_start", selected=commercial_finish_report.get("selected"), cached_candidate_analysis=bool(isinstance(commercial_finish_report.get("output"), dict) and commercial_finish_report.get("output")))
        immediate_candidate_analysis = commercial_finish_report.get("output") if isinstance(commercial_finish_report.get("output"), dict) and commercial_finish_report.get("output") else analyze_audio_fast_qc(commercial_candidate, sr, true_peak_oversample=_qc_oversample_factor())
        standard_problem = _candidate_problem_set(before, standard_final_reference, decision, finish_profile_for_guard, _safety_risk(before, standard_final_reference, decision), base_problem_set=[])
        candidate_problem = _candidate_problem_set(before, immediate_candidate_analysis, decision, finish_profile_for_guard, _safety_risk(before, immediate_candidate_analysis, decision), base_problem_set=list(standard_problem.get("problem_set", []) or []))
        v6299_arbiter = _v6299_final_perceptual_ab_arbiter(before, standard_final_reference, immediate_candidate_analysis, decision, candidate_problem)
        commercial_finish_report["v62_9_9_final_perceptual_ab_arbiter"] = v6299_arbiter
        commercial_finish_report["final_replacement_guard"] = {
            "schema_version": "busy_selected_candidate_final_replacement_guard_v8_5_3_21_plus_v62_9_9_perceptual_arbiter",
            "standard_problem_set": standard_problem,
            "candidate_problem_set": candidate_problem,
            "v62_9_9_final_perceptual_ab_arbiter": v6299_arbiter,
            "accepted": bool((not candidate_problem.get("critical_new_vs_base")) and not bool(v6299_arbiter.get("reject"))),
            "policy": "selected candidate may replace standard final only if it introduces no new critical problem set and passes the v62.9.9 perceptual/LRA/crest arbiter",
        }
        if candidate_problem.get("critical_new_vs_base") or bool(v6299_arbiter.get("reject")):
            commercial_selected = False
            commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
            commercial_finish_report["candidate_rejected_after_final_guard"] = True
            commercial_finish_report["candidate_rejection_reason"] = "v62_9_9_final_perceptual_ab_arbiter" if bool(v6299_arbiter.get("reject")) and not candidate_problem.get("critical_new_vs_base") else "new_unresolved_problem_set_after_final_guard"
            commercial_finish_report["candidate_rejection_problem_set"] = candidate_problem.get("critical_new_vs_base") or v6299_arbiter.get("reasons")
            final = standard_final_audio_for_fallback
            commercial_finish_report = _v6299_normalize_commercial_finish_after_guard(commercial_finish_report, standard_final_reference)
        else:
            commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = True
            final = commercial_candidate
    else:
        commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
        commercial_finish_report.setdefault("fallback_final_source", "standard_final_limiter_output")
    _debug(
        "engine_done_commercial_loudness_finish",
        active=commercial_finish_report.get("active"),
        target_met=commercial_finish_report.get("target_met"),
        selected=commercial_finish_report.get("selected"),
        reason=commercial_finish_report.get("reason"),
        output=(commercial_finish_report.get("output") or {}).get("integrated_lufs"),
        strategy_application_stage=commercial_finish_report.get("strategy_application_stage"),
    )
    try:
        final = np.nan_to_num(np.asarray(final, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
    # v63.1.6.2: the runtime cache must describe the actual delivered buffer,
    # not a stale commercial checkpoint that may have been rejected or only
    # partially selected.  This prevents v60/v61/QC from comparing against the
    # wrong authority.
    try:
        _cached_final_analysis_for_runtime = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
        commercial_finish_report["delivered_final_runtime_reference_analysis"] = _analysis_summary_for_loudness(_cached_final_analysis_for_runtime)
    except Exception:
        _cached_final_analysis_for_runtime = commercial_finish_report.get("output") if isinstance(commercial_finish_report, dict) and isinstance(commercial_finish_report.get("output"), dict) else {}

    # v8.5.3.34: final safety ceiling enforcement before selected-master lock.
    # Optimizer/candidate replacement can slightly move true peak after the first
    # limiter.  The final selected chain must still obey the locked intensity TP
    # ceiling before dither/export.
    final_safety_ceiling_enforcement = {
        "schema_version": "busy_final_safety_ceiling_enforcement_v8_5_3_34",
        "active": False,
        "target_dbtp": round(float(FINAL_TRUE_PEAK_CEILING_DB), 3),
        "policy": "hard TP ceiling is enforced before selected-master lock; no post-lock loudness/gain pass is allowed",
    }
    try:
        _pre_final_tp_analysis = _cached_final_analysis_for_runtime if isinstance(_cached_final_analysis_for_runtime, dict) and _cached_final_analysis_for_runtime else analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
        _pre_final_tp = float(_pre_final_tp_analysis.get("approx_true_peak_dbfs", _pre_final_tp_analysis.get("true_peak_dbfs", -99.0)) or -99.0)
        final_safety_ceiling_enforcement["pre_enforcement_true_peak_dbtp"] = round(_pre_final_tp, 3)
        if _pre_final_tp > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.03:
            final, _ceiling_gain_db = _normalize_to_true_peak(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=_qc_oversample_factor())
            final_safety_ceiling_enforcement.update({
                "active": True,
                "gain_db": round(float(_ceiling_gain_db), 3),
            })
            _post_final_tp_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
            final_safety_ceiling_enforcement["post_enforcement_true_peak_dbtp"] = round(float(_post_final_tp_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0), 3)
            _debug("engine_done_final_safety_ceiling_enforcement", **final_safety_ceiling_enforcement)
        else:
            final_safety_ceiling_enforcement["reason"] = "already_within_locked_true_peak_ceiling"
    except Exception as exc:
        final_safety_ceiling_enforcement.update({"active": False, "reason": "exception", "error": str(exc)[:300]})
        _debug("engine_error_final_safety_ceiling_enforcement", error=str(exc)[:300])

    _pre_lra_recovery_analysis = _cached_final_analysis_for_runtime if isinstance(_cached_final_analysis_for_runtime, dict) and _cached_final_analysis_for_runtime else analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    final, lra_recovery_stage = _maybe_apply_lra_recovery_stage(final, sr, before, _pre_lra_recovery_analysis, decision, mode, commercial_finish_report)
    _debug("engine_done_lra_recovery_stage", active=lra_recovery_stage.get("active"), applied=lra_recovery_stage.get("applied"), reason=lra_recovery_stage.get("reason"), candidate=lra_recovery_stage.get("candidate", {}))

    # v63.1.6.2: deterministic multistage limiter runs as a final-stage
    # architecture against the current delivered final reference, not the raw
    # pre-limiter checkpoint.  This keeps v60 from producing a quiet TP-only
    # candidate and lets its post-safety QC compare to the actual final buffer.
    if _env_bool("BUSY_V60_DML_FULL_ANALYSIS", False):
        try:
            _v60_standard_analysis = analyze_audio(final, sr)
        except Exception:
            _v60_standard_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    else:
        _v60_standard_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    _v60_standard_analysis = _v6362_ensure_lra_for_dml(_v60_standard_analysis, final, sr, source="standard_final_reference_before_dml")
    _v60_source = final
    _debug("engine_start_v60_deterministic_multistage_limiter", source="current_final_reference", source_shape=getattr(_v60_source, "shape", None), final_shape=getattr(final, "shape", None), cached_standard_analysis=bool(isinstance(_cached_final_analysis_for_runtime, dict) and _cached_final_analysis_for_runtime), standard_lra=_analysis_lra_lu(_v60_standard_analysis), dml_engine_version=V6362_DML_ENGINE_VERSION)
    _v60_candidate_audio, v60_deterministic_multistage_limiter = _v60_deterministic_multistage_limiter(
        _v60_source,
        sr,
        before,
        decision,
        _v60_standard_analysis,
        os_factor=os_factor,
        finish_report=commercial_finish_report,
    )
    if isinstance(v60_deterministic_multistage_limiter, dict):
        v60_deterministic_multistage_limiter["v63162_source_role"] = "current_final_reference_after_commercial_finish_and_final_safety"
        v60_deterministic_multistage_limiter["engine_version"] = V6362_DML_ENGINE_VERSION
        _v6362_contract = v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else (v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {}
        if isinstance(_v6362_contract, dict) and _v6362_contract:
            commercial_finish_report["v6362_dml_ownership_contract"] = _v6362_contract
    if isinstance(v60_deterministic_multistage_limiter, dict) and v60_deterministic_multistage_limiter.get("applied"):
        final = _v60_candidate_audio
        commercial_finish_report["selected_final_source"] = "v60_deterministic_multistage_limiter"
        commercial_finish_report["fallback_final_source"] = "v60_deterministic_multistage_limiter"
        commercial_finish_report["v60_dml_replaced_standard_final"] = True
    elif isinstance(v60_deterministic_multistage_limiter, dict):
        commercial_finish_report["v60_dml_preserved_existing_final"] = True
    _debug("engine_done_v60_deterministic_multistage_limiter", active=v60_deterministic_multistage_limiter.get("active"), accepted=v60_deterministic_multistage_limiter.get("accepted"), applied=v60_deterministic_multistage_limiter.get("applied"), reason=v60_deterministic_multistage_limiter.get("reason"))

    final, v6314_3_mono_translation_guard = _v6314_3_apply_mono_translation_guard(
        final,
        sr,
        before,
        decision,
        commercial_finish_report,
    )
    if isinstance(v6314_3_mono_translation_guard, dict):
        commercial_finish_report["v6314_3_mono_translation_guard"] = v6314_3_mono_translation_guard
        if v6314_3_mono_translation_guard.get("applied"):
            _cached_final_analysis_for_runtime = {}
    _debug(
        "engine_done_v6314_3_mono_translation_guard",
        active=v6314_3_mono_translation_guard.get("active") if isinstance(v6314_3_mono_translation_guard, dict) else None,
        applied=v6314_3_mono_translation_guard.get("applied") if isinstance(v6314_3_mono_translation_guard, dict) else None,
        before_delta=v6314_3_mono_translation_guard.get("before_stereo_minus_mono_lufs_db") if isinstance(v6314_3_mono_translation_guard, dict) else None,
        after_delta=v6314_3_mono_translation_guard.get("after_stereo_minus_mono_lufs_db") if isinstance(v6314_3_mono_translation_guard, dict) else None,
        reason=v6314_3_mono_translation_guard.get("reason") if isinstance(v6314_3_mono_translation_guard, dict) else None,
    )

    _pre_v6361_dtp_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    # Cache now matches the current post-DML final. If v63.6.1 applies audio,
    # the cache is cleared below so final analysis is recomputed.
    _cached_final_analysis_for_runtime = _pre_v6361_dtp_analysis
    final, v6361_drum_transient_punch = apply_drum_transient_punch_complete(
        final,
        sr,
        input_analysis=before if isinstance(before, dict) else {},
        current_analysis=_pre_v6361_dtp_analysis,
        decision=decision if isinstance(decision, dict) else {},
        mode=mode,
        true_peak_ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
        oversample=_qc_oversample_factor(),
        context={
            "stage": "single_stage_after_dml_and_mono_guard_before_dither",
            "commercial_finish_report": commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
            "v6362_dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "v60_deterministic_multistage_limiter": v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {},
            "v6314_3_mono_translation_guard": v6314_3_mono_translation_guard if 'v6314_3_mono_translation_guard' in locals() else {},
        },
        analyzer=lambda audio, rate: analyze_audio_fast_qc(audio, rate, true_peak_oversample=_qc_oversample_factor()),
    )
    if isinstance(v6361_drum_transient_punch, dict) and v6361_drum_transient_punch.get("applied"):
        _cached_final_analysis_for_runtime = {}
    if isinstance(commercial_finish_report, dict):
        commercial_finish_report["v6361_drum_transient_punch"] = v6361_drum_transient_punch
    _debug("engine_done_v6361_drum_transient_punch", active=v6361_drum_transient_punch.get("active"), applied=v6361_drum_transient_punch.get("applied"), reason=v6361_drum_transient_punch.get("reason"), selected_score=v6361_drum_transient_punch.get("selected_score"))

    # v64.0.4: Commercial/Maximum density push foundation.  This is a real
    # rendered-audio candidate stage, not shadow telemetry.  It runs after the
    # guarded v63.6.1 transient module and before dither/final analysis so the
    # downloadable WAV can actually occupy the -0.1 dBTP commercial ceiling.
    v6404_commercial_maximum_density = {
        "schema_version": "busy_commercial_maximum_density_v8_5_3_64_0_4",
        "engine_version": V6404_COMMERCIAL_DENSITY_ENGINE_VERSION,
        "enabled": False,
        "active": False,
        "applied": False,
        "reason": "not_run",
    }
    try:
        _pre_v6404_density_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
        final, v6404_commercial_maximum_density = apply_commercial_maximum_density_push(
            final,
            sr,
            before_analysis=before if isinstance(before, dict) else {},
            current_analysis=_pre_v6404_density_analysis,
            decision=decision if isinstance(decision, dict) else {},
            mode=mode,
            commercial_finish_report=commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
            true_peak_ceiling_db=-0.1 if str(mode or "").lower().find("safe") < 0 else FINAL_TRUE_PEAK_CEILING_DB,
            oversample=_qc_oversample_factor(),
            analyzer=lambda audio, rate: analyze_audio_fast_qc(audio, rate, true_peak_oversample=_qc_oversample_factor()),
        )
        if isinstance(v6404_commercial_maximum_density, dict) and v6404_commercial_maximum_density.get("applied"):
            _cached_final_analysis_for_runtime = v6404_commercial_maximum_density.get("selected_analysis") if isinstance(v6404_commercial_maximum_density.get("selected_analysis"), dict) else {}
            if isinstance(commercial_finish_report, dict):
                commercial_finish_report["selected_final_source"] = "v6404_commercial_maximum_density"
                commercial_finish_report["fallback_final_source"] = "v6404_commercial_maximum_density"
        elif isinstance(v6404_commercial_maximum_density, dict):
            _cached_final_analysis_for_runtime = _pre_v6404_density_analysis
        if isinstance(commercial_finish_report, dict):
            commercial_finish_report["v6404_commercial_maximum_density"] = {k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"}
        if isinstance(decision, dict):
            decision["v6404_commercial_maximum_density"] = {k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"}
        _debug(
            "engine_done_v6404_commercial_maximum_density",
            active=v6404_commercial_maximum_density.get("active") if isinstance(v6404_commercial_maximum_density, dict) else None,
            applied=v6404_commercial_maximum_density.get("applied") if isinstance(v6404_commercial_maximum_density, dict) else None,
            mode_policy=v6404_commercial_maximum_density.get("mode_policy") if isinstance(v6404_commercial_maximum_density, dict) else None,
            selected=v6404_commercial_maximum_density.get("selected_candidate_id") if isinstance(v6404_commercial_maximum_density, dict) else None,
            final_tp=v6404_commercial_maximum_density.get("final_true_peak_dbtp") if isinstance(v6404_commercial_maximum_density, dict) else None,
            under_pushed=v6404_commercial_maximum_density.get("under_pushed") if isinstance(v6404_commercial_maximum_density, dict) else None,
        )
    except Exception as exc:
        v6404_commercial_maximum_density = {
            "schema_version": "busy_commercial_maximum_density_v8_5_3_64_0_4",
            "engine_version": V6404_COMMERCIAL_DENSITY_ENGINE_VERSION,
            "enabled": True,
            "active": False,
            "applied": False,
            "reason": "exception_preserve_existing_final",
            "error": str(exc)[:300],
        }
        if isinstance(commercial_finish_report, dict):
            commercial_finish_report["v6404_commercial_maximum_density"] = v6404_commercial_maximum_density
        _debug("engine_error_v6404_commercial_maximum_density", error=str(exc)[:300])

    final, dither_report = _apply_output_dither(final, bit_depth=24)
    if dither_report.get("active"):
        _debug("engine_done_dither_noise_shaping", **dither_report)

    _debug("engine_start_final_analysis", runtime_safe=not _env_bool("BUSY_FINAL_ANALYSIS_FULL", False))
    _final_analysis_started_at = time.monotonic()
    final_analysis = _runtime_safe_final_analysis(final, sr, _cached_final_analysis_for_runtime if "_cached_final_analysis_for_runtime" in locals() else None)
    _debug("engine_done_final_analysis", elapsed_sec=round(max(0.0, time.monotonic() - _final_analysis_started_at), 3), schema=final_analysis.get("schema_version"), lufs=final_analysis.get("integrated_lufs"), true_peak=final_analysis.get("approx_true_peak_dbfs"))

    final_export_lock = {
        "schema_version": "busy_final_export_lock_v8_5_3_63_1_3_3_export_lock_not_applicable_fix",
        "enabled": _env_bool("BUSY_V48_2_2_FINAL_EXPORT_LOCK", True),
        "active": False,
        "accepted": True,
        "comparison_analyzer": "not_applicable_standard_final_or_no_selected_candidate",
        "mismatch_reasons": [],
        "buffer_identity_match": None,
        "identity_stage": "engine_return_buffer_consistency_guard",
        "policy": "selected commercial candidate full export analysis and buffer identity must match the final returned buffer before export; when no candidate replaced the standard final, the lock is not applicable and must not be reported as rejected.",
    }
    try:
        _selected_render_report = commercial_finish_report.get("selected_candidate_final_grade_render", {}) if isinstance(commercial_finish_report, dict) else {}
        _expected_selected, _expected_analysis_mode = _v56_selected_export_expected_analysis(_selected_render_report)
        _candidate_replaced = bool(commercial_finish_report.get("selected_candidate_replaced_standard_final_limiter")) if isinstance(commercial_finish_report, dict) else False
        if final_export_lock["enabled"] and _candidate_replaced and isinstance(_expected_selected, dict) and _expected_selected:
            _delta = _v48_2_2_export_lock_delta(_expected_selected, final_analysis)
            _mismatch = _v48_2_2_export_lock_mismatch(_delta)
            if final_analysis.get("final_analysis_runtime_safe") and _env_bool("BUSY_EXPORT_LOCK_RUNTIME_SAFE_NO_ROLLBACK", True):
                final_export_lock["runtime_safe_analysis_rollback_suppressed"] = bool(_mismatch)
                final_export_lock["runtime_safe_original_mismatch_reasons"] = list(_mismatch or [])
                _mismatch = []
            _returned_identity = _v56_audio_buffer_identity(
                final,
                sr,
                final_analysis,
                stage="engine_return_buffer_before_export_lock_decision",
                source="selected_candidate_current_final" if _candidate_replaced else "standard_final",
                candidate_label=str(commercial_finish_report.get("selected") or "") if isinstance(commercial_finish_report, dict) else None,
            )
            _selected_identity = _selected_render_report.get("buffer_identity") if isinstance(_selected_render_report, dict) and isinstance(_selected_render_report.get("buffer_identity"), dict) else {}
            _identity_match = (bool(_selected_identity and _returned_identity and _selected_identity.get("sha256") == _returned_identity.get("sha256")) if _selected_identity else None)
            final_export_lock.update({
                "schema_version": "busy_final_export_lock_v8_5_3_63_1_3_3_export_lock_not_applicable_fix",
                "active": True,
                "candidate_id": commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None,
                "delta": _delta,
                "mismatch_reasons": _mismatch,
                "accepted": not bool(_mismatch),
                "comparison_analyzer": _expected_analysis_mode,
                "selected_candidate_buffer_identity": _selected_identity,
                "returned_buffer_identity": _returned_identity,
                "buffer_identity_match": _identity_match,
                "false_mismatch_guard": {
                    "schema_version": "busy_v56_export_lock_false_mismatch_guard",
                    "active": _expected_analysis_mode == "full_analyze_audio",
                    "reason": "selected candidate expected analysis uses full analyze_audio to match final export analysis" if _expected_analysis_mode == "full_analyze_audio" else "legacy fast_qc expected analysis fallback",
                },
            })
            if isinstance(commercial_finish_report, dict):
                commercial_finish_report["v56_export_identity_lock"] = {
                    "schema_version": "busy_v56_candidate_handoff_export_identity_lock",
                    "active": True,
                    "candidate_id": commercial_finish_report.get("selected"),
                    "comparison_analyzer": _expected_analysis_mode,
                    "accepted": not bool(_mismatch),
                    "mismatch_reasons": _mismatch,
                    "selected_sha256_16": _selected_identity.get("sha256_16") if isinstance(_selected_identity, dict) else None,
                    "returned_sha256_16": _returned_identity.get("sha256_16") if isinstance(_returned_identity, dict) else None,
                    "buffer_identity_match": _identity_match,
                    "policy": "same selected-candidate lineage must use comparable analysis and tracked buffer identity before export.",
                }
            if _mismatch and _env_bool("BUSY_V48_2_2_EXPORT_LOCK_ROLLBACK", True):
                final_export_lock["rollback_to_standard_final"] = True
                final_export_lock["rollback_reason"] = "selected_candidate_output_analysis_mismatch"
                try:
                    final = np.nan_to_num(np.asarray(standard_final_audio_for_fallback, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    final, _lock_tp_gain, _lock_tp_report = _attenuate_to_true_peak_ceiling(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=_qc_oversample_factor())
                    final_export_lock["fallback_true_peak_normalize_gain_db"] = round(float(_lock_tp_gain), 3)
                    final_export_lock["fallback_true_peak_ceiling_attenuation"] = _lock_tp_report
                    final, _lock_dither_report = _apply_output_dither(final, bit_depth=24)
                    final_export_lock["fallback_dither_report"] = _lock_dither_report
                    final_analysis = _runtime_safe_final_analysis(final, sr, None)
                    final_export_lock["fallback_output_analysis"] = _analysis_summary_for_loudness(final_analysis)
                    if isinstance(commercial_finish_report, dict):
                        commercial_finish_report["final_export_lock_rejected_selected_candidate"] = True
                        commercial_finish_report["final_export_lock_rejection_reasons"] = _mismatch
                        commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
                        commercial_finish_report["fallback_final_source"] = "standard_final_after_export_lock"
                except Exception as exc:
                    final_export_lock["rollback_error"] = str(exc)[:300]
        else:
            final_export_lock.update({
                "active": False,
                "accepted": True,
                "reason": "no_selected_commercial_candidate_or_expected_analysis_missing",
                "comparison_analyzer": "not_applicable_standard_final_or_no_selected_candidate",
                "mismatch_reasons": [],
                "buffer_identity_match": None,
                "not_applicable_guard": {
                    "schema_version": "busy_export_lock_not_applicable_guard_v8_5_3_63_1_3_3",
                    "candidate_replaced_standard_final": bool(_candidate_replaced),
                    "expected_analysis_available": bool(isinstance(_expected_selected, dict) and _expected_selected),
                    "policy": "An inactive export lock is not a rejection; standard-final or no-expected-analysis paths are marked accepted-but-not-applicable for debug/Supabase clarity.",
                },
            })
    except Exception as exc:
        final_export_lock.update({"active": False, "reason": "exception", "error": str(exc)[:300]})

    _debug("engine_done_final_export_lock", active=final_export_lock.get("active"), accepted=final_export_lock.get("accepted"), reason=final_export_lock.get("reason"), rollback=final_export_lock.get("rollback_to_standard_final"))
    commercial_finish_report = _v62_8_2_apply_measurement_authority_runtime_safe(commercial_finish_report, final_analysis)
    _debug("engine_done_measurement_authority_finish")
    reasons = _safety_risk(before, final_analysis, decision)
    _debug("engine_done_safety_risk", count=len(reasons or []))
    stereo_qc = _stereo_qc_summary(final_analysis)
    _debug("engine_done_stereo_qc_build")
    ab_delta_qc = _ab_delta_qc(before, final_analysis)
    _debug("engine_done_ab_delta_qc_build")
    lra_guard = _lra_guard_summary(before, final_analysis, decision, mode)
    _debug("engine_done_lra_guard_build")
    codec_preview_guard = _codec_preview_guard(before, final_analysis)
    _debug("engine_done_codec_preview_guard_build")
    playback_translation = _playback_translation_report(before, final_analysis)
    _debug("engine_done_playback_translation_build")
    reasons = sorted(set(
        reasons
        + stereo_qc.get("warnings", [])
        + ab_delta_qc.get("warnings", [])
        + lra_guard.get("warnings", [])
        + codec_preview_guard.get("warnings", [])
        + playback_translation.get("warnings", [])
    ))
    adaptive_input_relative_qc_governance = _adaptive_input_relative_qc_governance(
        before, final_analysis, decision, mode, lra_guard, reasons
    )
    try:
        _wr = adaptive_input_relative_qc_governance.get("warning_reclassification", {}) if isinstance(adaptive_input_relative_qc_governance, dict) else {}
        if isinstance(_wr, dict) and _wr.get("adjusted_warnings"):
            reasons = [str(x) for x in _wr.get("adjusted_warnings", [])]
            # v62.9.9.3.1: preserve raw LRA-guard output while making the
            # displayed LRA warnings match adaptive input-relative governance.
            _raw_lra_warnings = [str(x) for x in (lra_guard.get("warnings", []) or [])]
            lra_guard["raw_warnings_before_adaptive_input_relative_qc"] = _raw_lra_warnings
            lra_guard["adaptive_input_relative_reclassification"] = _wr
            _removed = set(str(x) for x in (_wr.get("removed_warnings", []) or []))
            _added_lra = [
                str(x) for x in (_wr.get("added_warnings", []) or [])
                if str(x) in {
                    "source_lra_already_flat",
                    "lra_flat_but_input_relative_loss_small",
                    "source_flat_plus_additional_lra_loss",
                    "mastering_induced_lra_collapse",
                    "lra_low_input_unavailable",
                }
            ]
            _adj_lra = [w for w in _raw_lra_warnings if w not in _removed]
            for _w in _added_lra:
                if _w not in _adj_lra:
                    _adj_lra.append(_w)
            lra_guard["warnings"] = sorted(_adj_lra)
            lra_guard["adaptive_adjusted_lra_warnings"] = sorted(_adj_lra)
    except Exception:
        pass
    lra_warning_semantics = _v6313_3_lra_warning_semantics(lra_guard, adaptive_input_relative_qc_governance, reasons)
    if isinstance(lra_guard, dict):
        lra_guard["warning_semantics_v6313_3"] = lra_warning_semantics
    _debug("engine_done_adaptive_input_relative_qc", active=adaptive_input_relative_qc_governance.get("active") if isinstance(adaptive_input_relative_qc_governance, dict) else None, lra=((adaptive_input_relative_qc_governance.get("lra") or {}) if isinstance(adaptive_input_relative_qc_governance, dict) else {}), warnings=reasons)
    _debug("engine_done_stereo_qc", **stereo_qc)
    _debug("engine_done_lra_guard", **lra_guard)
    _debug("engine_done_codec_preview_guard", **codec_preview_guard)
    _debug("engine_done_playback_translation", overall_score=playback_translation.get("overall_score"), warnings=playback_translation.get("warnings", []))
    _debug("engine_done_ab_delta_qc", warnings=ab_delta_qc.get("warnings", []), deltas=ab_delta_qc.get("deltas", {}))
    _debug("engine_done_final_analysis", reasons=reasons, lufs=final_analysis.get("integrated_lufs"), lra=final_analysis.get("lra_lu"), true_peak=final_analysis.get("approx_true_peak_dbfs"), crest=final_analysis.get("crest_factor_db"), correlation=final_analysis.get("correlation"))
    _v57_2_attach_delivered_final_damage_gate(commercial_finish_report, before, final_analysis, decision, mode, lra_guard, reasons)

    perceptual_quality_score = calculate_perceptual_quality_score(before, final_analysis)
    if perceptual_quality_score.get("warnings"):
        reasons = sorted(set(reasons + [str(w) for w in perceptual_quality_score.get("warnings", [])]))
    _debug("engine_done_perceptual_quality_score", composite_pqs=perceptual_quality_score.get("composite_pqs"), warnings=perceptual_quality_score.get("warnings", []))

    # Prediction-vs-measurement feedback for calibrating the virtual solver.
    strategy_for_qc = commercial_finish_report.get("final_push_strategy", {}) if isinstance(commercial_finish_report.get("final_push_strategy", {}), dict) else {}
    selected_candidate_for_qc = None
    for _cand in commercial_finish_report.get("candidates", []) or []:
        if isinstance(_cand, dict) and _cand.get("name") == commercial_finish_report.get("selected"):
            selected_candidate_for_qc = _cand
            break
    predicted_lufs = strategy_for_qc.get("expected_lufs_after_push_est")
    predicted_tp = strategy_for_qc.get("recommended_true_peak_ceiling_db")
    measured_lufs = final_analysis.get("integrated_lufs")
    measured_tp = final_analysis.get("approx_true_peak_dbfs")
    measured_crest = final_analysis.get("crest_factor_db")
    actual_vs_predicted_qc = {
        "schema_version": "busy_actual_vs_predicted_qc_v8_5_3_21",
        "prediction_source": "virtual_hot_commercial_solver.final_push_strategy",
        "actual_source": "final_output_analysis_after_strategy_application",
        "selected_candidate": commercial_finish_report.get("selected"),
        "strategy_application_stage": commercial_finish_report.get("strategy_application_stage"),
        "predicted_lufs": predicted_lufs,
        "measured_lufs": measured_lufs,
        "lufs_error_db": round(float(measured_lufs) - float(predicted_lufs), 3) if predicted_lufs is not None and measured_lufs is not None else None,
        "predicted_true_peak_db": predicted_tp,
        "measured_true_peak_db": measured_tp,
        "true_peak_error_db": round(float(measured_tp) - float(predicted_tp), 3) if predicted_tp is not None and measured_tp is not None else None,
        "measured_crest_db": measured_crest,
        "selected_candidate_analysis": (selected_candidate_for_qc or {}).get("analysis", {}) if isinstance(selected_candidate_for_qc, dict) else {},
        "calibration_hint": None,
    }
    try:
        le = actual_vs_predicted_qc.get("lufs_error_db")
        tpe = actual_vs_predicted_qc.get("true_peak_error_db")
        hints = []
        if le is not None and float(le) < -0.6:
            hints.append("virtual_lufs_efficiency_overestimated")
        if le is not None and float(le) > 0.6:
            hints.append("virtual_lufs_efficiency_underestimated")
        if tpe is not None and float(tpe) > 0.25:
            hints.append("true_peak_proxy_underestimated")
        if commercial_finish_report.get("selected") == "base":
            hints.append("no_candidate_selected_or_strategy_application_failed")
        actual_vs_predicted_qc["calibration_hint"] = hints
    except Exception:
        actual_vs_predicted_qc["calibration_hint"] = ["calibration_hint_error"]

    playback_translation_auditor = build_playback_translation_auditor(before, final_analysis, playback_translation)
    post_master_qc_ledger = build_post_master_qc_ledger(before, final_analysis, decision, {
        "remaining_safety_warnings": reasons,
        "rollback_used": rollback_used,
        "rollback_attempts": rollback_attempts,
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
        "commercial_loudness_finish": commercial_finish_report,
        "virtual_hot_commercial_solver": commercial_finish_report.get("virtual_hot_commercial_solver", {}),
        "blocker_capacity_map": commercial_finish_report.get("blocker_capacity_map", {}),
        "final_push_strategy": commercial_finish_report.get("final_push_strategy", {}),
        "musical_role_identity_map": commercial_finish_report.get("musical_role_identity_map", {}),
        "virtual_premaster_strategy": commercial_finish_report.get("virtual_premaster_strategy", {}),
        "multi_agent_virtual_premaster_strategy_council": commercial_finish_report.get("multi_agent_virtual_premaster_strategy_council", {}),
        "actual_vs_predicted_qc": actual_vs_predicted_qc,
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
        "lra_warning_semantics": lra_warning_semantics if 'lra_warning_semantics' in locals() else {},
        "adaptive_input_relative_qc_governance": adaptive_input_relative_qc_governance,
        "codec_preview_guard": codec_preview_guard,
        "playback_translation_auditor": playback_translation_auditor,
        "final_safety_ceiling_enforcement": final_safety_ceiling_enforcement,
        "lra_recovery_stage": lra_recovery_stage,
        "v6361_drum_transient_punch": v6361_drum_transient_punch if 'v6361_drum_transient_punch' in locals() else {},
        "drum_transient_punch": v6361_drum_transient_punch if 'v6361_drum_transient_punch' in locals() else {},
        "v6404_commercial_maximum_density": ({k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"} if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else {}),
        "commercial_maximum_density": ({k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"} if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else {}),
        "v6314_3_mono_translation_guard": v6314_3_mono_translation_guard if 'v6314_3_mono_translation_guard' in locals() else {},
        "perceptual_quality_score": perceptual_quality_score,
    })
    final_master_confidence = build_final_master_confidence(
        post_master_qc_ledger,
        playback_translation_auditor,
        codec_preview_guard,
        before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        final_analysis,
    )
    ai_residue_architecture_mapper = build_ai_residue_architecture_report(
        before,
        final_analysis,
        decision,
        {
            "loudness_push_governor": governor_report,
            "commercial_loudness_prep": commercial_loudness_prep,
            "experimental_residue_light_control": experimental_residue_light_control,
            "adaptive_residue_light_control": experimental_residue_light_control,
            "residue_pre_clean_stage": residue_pre_clean_stage,
            "commercial_loudness_finish": commercial_finish_report,
            "virtual_hot_commercial_solver": commercial_finish_report.get("virtual_hot_commercial_solver", {}),
            "blocker_capacity_map": commercial_finish_report.get("blocker_capacity_map", {}),
            "final_push_strategy": commercial_finish_report.get("final_push_strategy", {}),
            "musical_role_identity_map": commercial_finish_report.get("musical_role_identity_map", {}),
            "virtual_premaster_strategy": commercial_finish_report.get("virtual_premaster_strategy", {}),
            "processing_report": final_proc,
            "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        },
        post_master_qc_ledger,
        playback_translation_auditor,
    )
    _debug("engine_done_post_master_qc_ledger", confidence=post_master_qc_ledger.get("confidence"), decisions=post_master_qc_ledger.get("decisions"), warnings=[w.get("code") for w in post_master_qc_ledger.get("warning_details", []) if isinstance(w, dict)])
    _debug("engine_done_final_master_confidence", confidence=final_master_confidence.get("confidence"), decisions=final_master_confidence.get("decisions"))
    _debug("engine_done_ai_residue_architecture_mapper", confidence=ai_residue_architecture_mapper.get("confidence"), action_level=(ai_residue_architecture_mapper.get("final_action_policy") or {}).get("action_level"), candidates=len(((ai_residue_architecture_mapper.get("shadow_residue_attenuation") or {}).get("candidate_actions") or [])))

    final_master_highlight_review = _run_gemini_final_master_highlight_review_best_effort(
        final,
        sr,
        before if isinstance(before, dict) else {},
        final_analysis if isinstance(final_analysis, dict) else {},
        decision if isinstance(decision, dict) else {},
        commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
        final_export_lock if isinstance(final_export_lock, dict) else {},
        mode,
    )
    final_master_highlight_review_public = {k: v for k, v in final_master_highlight_review.items() if k not in {"raw_text"}}
    if isinstance(commercial_finish_report, dict):
        commercial_finish_report["final_master_highlight_review"] = final_master_highlight_review_public
    _debug(
        "engine_done_gemini_final_master_highlight_review",
        enabled=final_master_highlight_review_public.get("enabled"),
        available=final_master_highlight_review_public.get("available"),
        verdict=final_master_highlight_review_public.get("verdict"),
        commercial_readiness=final_master_highlight_review_public.get("commercial_readiness"),
        reason=final_master_highlight_review_public.get("reason"),
    )

    report = {
        "engine_version": V6362_MASTERING_ENGINE_VERSION,
        "mastering_report_schema_version": "busy_master_report_v8_5_3_64_0_4_commercial_maximum_density_push_foundation",
        "runtime_intensity_lock": runtime_intensity_lock,
        "optimized_pipeline_architecture": build_pipeline_architecture_state(pre_analysis or before if isinstance(pre_analysis or before, dict) else {}, decision, final_intensity=(runtime_intensity_lock or {}).get("final_intensity")),
        "render_budget_policy": render_budget_policy((runtime_intensity_lock or {}).get("final_intensity")),
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": original_before,
        "residue_pre_clean_stage": residue_pre_clean_stage,
        "output_analysis": final_analysis,
        "working_stage_gain_db": round(stage_gain_db, 3),
        "final_true_peak_gain_db": round(final_true_peak_gain_db, 3),
        "true_peak_ceiling_attenuation": true_peak_ceiling_attenuation if isinstance(locals().get("true_peak_ceiling_attenuation"), dict) else {},
        "final_safety_ceiling_enforcement": final_safety_ceiling_enforcement,
        "final_export_lock": final_export_lock,
        "final_true_peak_target_dbtp": FINAL_TRUE_PEAK_CEILING_DB,
        "requested_oversample_factor": adaptive_quality_report.get("requested_oversample"),
        "oversample_factor": os_factor,
        "adaptive_quality": adaptive_quality_report,
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
        "v58_spectral_conditioning_plan": decision.get("v58_spectral_conditioning_plan", {}) if isinstance(decision, dict) else {},
        "v58_spectral_conditioning_render_report": (final_proc or {}).get("v58_spectral_conditioning", {}) if isinstance(final_proc, dict) else {},
        "v59_lf_crest_source_conditioning_plan": decision.get("v59_lf_crest_source_conditioning_plan", {}) if isinstance(decision, dict) else {},
        "v59_lf_crest_source_conditioning_render_report": (final_proc or {}).get("v59_lf_crest_source_conditioning", {}) if isinstance(final_proc, dict) else {},
        "axis_ownership_ledger": {
            "v58": ((final_proc or {}).get("v58_spectral_conditioning", {}) or {}).get("axis_ownership_ledger", {}) if isinstance(final_proc, dict) else {},
            "v59": ((final_proc or {}).get("v59_lf_crest_source_conditioning", {}) or {}).get("axis_ownership_ledger", {}) if isinstance(final_proc, dict) else {},
            "policy": "Top-level debug mirror; authoritative ledgers remain inside v58/v59 render reports.",
        },
        "v62_8_mastering_blocker_preparation": decision.get("v62_8_mastering_blocker_preparation", {}) if isinstance(decision, dict) else {},
        "mix_conditioning_plan": decision.get("mix_conditioning_plan", {}) if isinstance(decision, dict) else {},
        "lgcca_influence_routing": decision.get("lgcca_influence_routing", {}) if isinstance(decision, dict) else {},
        "lgcca_deep_wiring": decision.get("lgcca_deep_wiring", {}) if isinstance(decision, dict) else {},
        "action_governance": decision.get("action_governance", {}) if isinstance(decision, dict) else {},
        "existing_process_research_governance": decision.get("existing_process_research_governance", {}) if isinstance(decision, dict) else {},
        "v62_9_9_research_governance": decision.get("v62_9_9_research_governance", {}) if isinstance(decision, dict) else {},
        "microdynamics_lra_shape_recovery": (decision.get("v62_9_9_research_governance", {}) or {}).get("microdynamics_lra_shape_recovery", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "section_aware_spectral_consistency": (decision.get("v62_9_9_research_governance", {}) or {}).get("section_aware_spectral_consistency", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "musical_texture_preservation_gate": (decision.get("v62_9_9_research_governance", {}) or {}).get("musical_texture_preservation_gate", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "translation_aware_low_end_governor": (decision.get("v62_9_9_research_governance", {}) or {}).get("translation_aware_low_end_governor", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "ai_call_governance": decision.get("ai_call_governance", {}) if isinstance(decision, dict) else {},
        "loudness_feasibility_predictor": decision.get("loudness_feasibility_predictor", {}) if isinstance(decision, dict) else {},
        "perceptual_quality_score": perceptual_quality_score,
        "mix_conditioning_render_report": (final_proc or {}).get("mix_conditioning", {}) if isinstance(final_proc, dict) else {},
        "adaptive_residue_light_control": experimental_residue_light_control,
        "experimental_residue_light_control": experimental_residue_light_control,
        "adaptive_hot_commercial": {
            "enabled": bool(_experimental_hot_commercial_enabled(mode)),
            "auto_residue_control_enabled": bool(_env_bool("BUSY_ADAPTIVE_RESIDUE_CONTROL", True)),
            "target_lufs_env_override": os.environ.get("BUSY_EXPERIMENTAL_HOT_TARGET_LUFS"),
            "strength": _env_float("BUSY_ADAPTIVE_RESIDUE_STRENGTH", _env_float("BUSY_AI_RESIDUE_DIRECT_STRENGTH", 1.0, 0.20, 2.0), 0.20, 2.0),
            "policy": "hot commercial candidate testing after any one-time pre-master residue cleanup; no repeated post-master trimming",
        },
        "experimental_hot_commercial": {
            "enabled": bool(_experimental_hot_commercial_enabled(mode)),
            "compat_alias": True,
            "policy": "compat alias; see adaptive_hot_commercial",
        },
        "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
        "lra_warning_semantics": lra_warning_semantics if 'lra_warning_semantics' in locals() else {},
        "adaptive_input_relative_qc_governance": adaptive_input_relative_qc_governance,
        "lra_recovery_stage": lra_recovery_stage,
        "v6361_drum_transient_punch": v6361_drum_transient_punch if 'v6361_drum_transient_punch' in locals() else {},
        "drum_transient_punch": v6361_drum_transient_punch if 'v6361_drum_transient_punch' in locals() else {},
        "v6404_commercial_maximum_density": ({k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"} if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else {}),
        "commercial_maximum_density": ({k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"} if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else {}),
        "v6314_3_mono_translation_guard": v6314_3_mono_translation_guard if 'v6314_3_mono_translation_guard' in locals() else {},
        "mastering_ownership": _v6351_select_report_ownership(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "commercial_push_safety_gate": _v6351_select_report_gate(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "selected_loudness_class": (_v6351_select_report_ownership(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}) or {}).get("selected_loudness_class"),
        "push_authority": (_v6351_select_report_ownership(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}) or {}).get("push_authority"),
        "v6362_dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "v60_deterministic_multistage_limiter": v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {},
        "v61_render_measured_feedback_optimizer": v61_feedback_optimizer if 'v61_feedback_optimizer' in locals() else {"enabled": False, "active": False, "attempted": False, "reason": "missing_from_single_stage_path"},
        "v6316_true_pre_limiter_handoff": v6316_true_pre_limiter_handoff if 'v6316_true_pre_limiter_handoff' in locals() else {},
        "multi_pass_optimizer": optimizer_report,
        "commercial_loudness_finish": commercial_finish_report,
        "final_perceptual_ab_arbiter": (commercial_finish_report.get("v62_9_9_final_perceptual_ab_arbiter") if isinstance(commercial_finish_report, dict) else None) or (((decision.get("v62_9_9_research_governance", {}) or {}).get("final_perceptual_ab_arbiter_policy", {})) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {}),
        "selected_candidate_final_replacement_guard": (commercial_finish_report.get("final_replacement_guard") if isinstance(commercial_finish_report, dict) else {}),
        "final_master_highlight_review": final_master_highlight_review_public,
        "gemini_final_master_highlight_review": final_master_highlight_review_public,
        "virtual_hot_commercial_solver": commercial_finish_report.get("virtual_hot_commercial_solver", {}),
        "representative_section_map": commercial_finish_report.get("representative_section_map", {}),
        "virtual_limiter_sweep": (commercial_finish_report.get("virtual_hot_commercial_solver", {}) or {}).get("virtual_limiter_sweep", {}) if isinstance(commercial_finish_report.get("virtual_hot_commercial_solver", {}), dict) else {},
        "blocker_capacity_map": commercial_finish_report.get("blocker_capacity_map", {}),
        "blocker_solution_plan": commercial_finish_report.get("blocker_solution_plan", {}),
        "final_push_strategy": commercial_finish_report.get("final_push_strategy", {}),
        "musical_role_identity_map": commercial_finish_report.get("musical_role_identity_map", {}),
        "virtual_premaster_strategy": commercial_finish_report.get("virtual_premaster_strategy", {}),
        "commercial_loudness_result": {
            "target_lufs": commercial_finish_report.get("target_lufs"),
            "commercial_floor_lufs": commercial_finish_report.get("commercial_floor_lufs"),
            "final_lufs": final_analysis.get("integrated_lufs"),
            "final_lufs_authority": commercial_finish_report.get("final_lufs_authority"),
            "final_lufs_stereo": commercial_finish_report.get("final_lufs_stereo"),
            "final_lufs_mono_downmix": commercial_finish_report.get("final_lufs_mono_downmix"),
            "target_met": commercial_finish_report.get("target_met"),
            "target_reached": commercial_finish_report.get("target_reached"),
            "retry_attempted": commercial_finish_report.get("retry_attempted"),
            "reason": commercial_finish_report.get("reason"),
            "retry_blocked_by": commercial_finish_report.get("retry_blocked_by"),
            "policy": "commercial loudness floor remains distinct from target_reached; v64.0.4 Commercial/Maximum can still render/select a separate actual density-push WAV candidate with warning advisories instead of silently stopping at floor",
            "v6404_selected_candidate_id": (v6404_commercial_maximum_density.get("selected_candidate_id") if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else None),
            "v6404_true_peak_target_honored": (v6404_commercial_maximum_density.get("true_peak_target_honored") if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else None),
        },
        "codec_preview_guard": codec_preview_guard,
        "playback_translation": playback_translation,
        "playback_translation_auditor": playback_translation_auditor,
        "post_master_qc_explainer": post_master_qc_ledger,
        "rollback_ledger": post_master_qc_ledger,
        "final_master_confidence": final_master_confidence,
        "ai_residue_architecture_mapper": ai_residue_architecture_mapper,
        "ai_sound_architecture_analysis": ai_residue_architecture_mapper.get("ai_sound_architecture_analysis", {}),
        "residue_component_separation": ai_residue_architecture_mapper.get("residue_component_separation", {}),
        "musicality_protection_map": ai_residue_architecture_mapper.get("musicality_protection_map", {}),
        "shadow_residue_attenuation": ai_residue_architecture_mapper.get("shadow_residue_attenuation", {}),
        "direct_residue_action_gate": ai_residue_architecture_mapper.get("direct_residue_action_gate", {}),
        "ai_residue_final_user_summary": ai_residue_architecture_mapper.get("final_user_summary", {}),
        "dither_noise_shaping": dither_report,
        "rollback_used": rollback_used,
        "rollback_attempts": rollback_attempts,
        "remaining_safety_warnings": reasons,
        "risk_metrics": _risk_metrics(before, final_analysis),
        "character_scale_initial": round(character_scale, 3),
        "genre_chain": decision.get("genre_chain", {}),
        "processing_report": final_proc,
        "decision_summary": {
            "detected_style": decision.get("detected_style") or decision.get("genre_id"),
            "selected_profile": decision.get("selected_profile"),
            "style_confidence": decision.get("style_confidence"),
            "boldness_level": decision.get("boldness_level", "auto"),
            "profile_blend": decision.get("profile_blend"),
            "mastering_intent": decision.get("mastering_intent"),
            "reference_v7": decision.get("reference_v7"),
            "detailed_genre_decision": decision.get("detailed_genre_decision"),
            "frequency_band_centrifuge_summary": (before.get("frequency_band_centrifuge") or {}).get("global_summary", {}) if isinstance(before, dict) and isinstance(before.get("frequency_band_centrifuge"), dict) else {},
        },
    }
    report = _v62_9_4_1_attach_top_level_final_metric_aliases(report, final_analysis, commercial_finish_report)
    report = _v6314_3_attach_handoff_qc_semantics(report, decision)
    report = _v62_9_9_3_2_reconcile_adaptive_input_relative_qc_report(report, decision, mode)
    report = _v6314_4_finalize_floor_tolerance_and_handoff_markers(report, decision)
    report = _v6362_8_finalize_report_authority(report, decision)
    _debug("engine_done", output_lufs=final_analysis.get("integrated_lufs"), output_true_peak=final_analysis.get("approx_true_peak_dbfs"), oversample_factor=os_factor, adaptive_quality=adaptive_quality_report)
    return final, report




def prepare_pre_limiter_stage(
    y: np.ndarray,
    sr: int,
    decision: dict[str, Any],
    mode: str = "Auto Commercial Master",
    pre_analysis: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Stage A-2a: prepare governed decision and working-level audio.

    This deliberately stops before the heavier render/rollback pass.  The caller
    can write the returned staged float audio to a checkpoint and let the worker
    exit before Stage A-2b continues.
    """
    runtime_intensity_lock = _refresh_runtime_mastering_policy(decision)
    _debug("engine_start_pre_limiter_prepare", sr=sr, shape=getattr(y, "shape", None), dtype=str(getattr(y, "dtype", "")), mode=mode, runtime_intensity_lock=runtime_intensity_lock)
    if pre_analysis is not None:
        before = pre_analysis
        _debug("engine_done_input_analysis", reused_precomputed=True, lufs=before.get("integrated_lufs"), true_peak=before.get("approx_true_peak_dbfs"), crest=before.get("crest_factor_db"), correlation=before.get("correlation"))
    else:
        before = analyze_audio(y, sr)
        _debug("engine_done_input_analysis", reused_precomputed=False, lufs=before.get("integrated_lufs"), true_peak=before.get("approx_true_peak_dbfs"), crest=before.get("crest_factor_db"), correlation=before.get("correlation"))

    original_before = before
    y, residue_pre_clean_stage, before = _apply_residue_pre_clean_once(y, sr, before, decision, mode)
    _debug(
        "engine_done_residue_pre_clean_stage",
        active=residue_pre_clean_stage.get("active"),
        bypassed=residue_pre_clean_stage.get("bypassed"),
        reason=residue_pre_clean_stage.get("reason"),
        action_count=len(residue_pre_clean_stage.get("actions") or []),
    )

    governed_decision, governor_report = _loudness_push_governor(before, decision, mode)
    render_decision = governed_decision
    _debug("engine_done_loudness_push_governor", **governor_report)
    render_decision, spatial_fx_mastering = apply_spatial_fx_mastering_plan(render_decision, before)
    _debug("engine_done_spatial_fx_mastering_plan", active=spatial_fx_mastering.get("active"), risk_score=spatial_fx_mastering.get("risk_score"), dynamic_moves=spatial_fx_mastering.get("dynamic_move_count"), static_moves=spatial_fx_mastering.get("static_move_count"), notes=spatial_fx_mastering.get("notes"))
    render_decision, commercial_loudness_prep = _attach_commercial_loudness_prep(before, render_decision, mode)
    _debug("engine_done_commercial_loudness_prep_plan", **{k: v for k, v in commercial_loudness_prep.items() if k not in {"static_moves", "dynamic_moves"}})
    render_decision, v58_spectral_conditioning_plan = _attach_v58_spectral_conditioning(before, render_decision, mode)
    _debug("engine_done_v58_spectral_conditioning_plan", active=v58_spectral_conditioning_plan.get("active"), reasons=v58_spectral_conditioning_plan.get("reasons"), static_moves=v58_spectral_conditioning_plan.get("static_move_count"), dynamic_moves=v58_spectral_conditioning_plan.get("dynamic_move_count"))
    render_decision, v59_lf_crest_source_conditioning_plan = _attach_v59_lf_crest_source_conditioning(before, render_decision, mode)
    _debug("engine_done_v59_lf_crest_source_conditioning_plan", active=v59_lf_crest_source_conditioning_plan.get("active"), render_affecting=v59_lf_crest_source_conditioning_plan.get("render_affecting"), reasons=v59_lf_crest_source_conditioning_plan.get("reasons"), dynamic_moves=v59_lf_crest_source_conditioning_plan.get("dynamic_move_count"), time_actions=v59_lf_crest_source_conditioning_plan.get("time_domain_action_count"), stereo_side_actions=v59_lf_crest_source_conditioning_plan.get("stereo_side_action_count"))
    render_decision, v62_8_mastering_blocker_preparation = _attach_v62_8_mastering_blocker_preparation(before, render_decision, mode)
    _debug("engine_done_v62_8_mastering_blocker_preparation", active=v62_8_mastering_blocker_preparation.get("active"), actions=v62_8_mastering_blocker_preparation.get("action_count"), families=v62_8_mastering_blocker_preparation.get("micro_ab_candidate_families"), decision=(v62_8_mastering_blocker_preparation.get("commercial_loudness_feasibility") or {}).get("decision"))

    staged, stage_gain_db = _stage_to_working_lufs(y, sr, WORKING_STAGE_LUFS)
    character_scale = _boldness_scale(render_decision, mode)
    _debug("engine_done_working_stage", stage_gain_db=round(stage_gain_db, 3), character_scale=round(character_scale, 3), staged_shape=getattr(staged, "shape", None))

    prepare_report = {
        "engine_version": "busy_auto_mastering_v8_5_3_40_stage_a2a_mix_conditioning_prepare",
        "stage": "pre_limiter_prepare",
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": original_before,
        "residue_pre_clean_stage": residue_pre_clean_stage,
        "working_stage_gain_db": round(stage_gain_db, 3),
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
        "v58_spectral_conditioning_plan": render_decision.get("v58_spectral_conditioning_plan", {}) if isinstance(render_decision, dict) else {},
        "v59_lf_crest_source_conditioning_plan": render_decision.get("v59_lf_crest_source_conditioning_plan", {}) if isinstance(render_decision, dict) else {},
        "v62_8_mastering_blocker_preparation": render_decision.get("v62_8_mastering_blocker_preparation", {}) if isinstance(render_decision, dict) else {},
        "mix_conditioning_plan": render_decision.get("mix_conditioning_plan", {}) if isinstance(render_decision, dict) else {},
        "lgcca_influence_routing": render_decision.get("lgcca_influence_routing", {}) if isinstance(render_decision, dict) else {},
        "lgcca_deep_wiring": render_decision.get("lgcca_deep_wiring", {}) if isinstance(render_decision, dict) else {},
        "action_governance": render_decision.get("action_governance", {}) if isinstance(render_decision, dict) else {},
        "ai_call_governance": render_decision.get("ai_call_governance", {}) if isinstance(render_decision, dict) else {},
        "loudness_feasibility_predictor": render_decision.get("loudness_feasibility_predictor", {}) if isinstance(render_decision, dict) else {},
        "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        "spatial_fx_mastering": spatial_fx_mastering,
        "character_scale_initial": round(character_scale, 3),
        "genre_chain": render_decision.get("genre_chain", {}),
        "decision_summary": {
            "detected_style": render_decision.get("detected_style") or render_decision.get("genre_id"),
            "selected_profile": render_decision.get("selected_profile"),
            "style_confidence": render_decision.get("style_confidence"),
            "boldness_level": render_decision.get("boldness_level", "auto"),
            "profile_blend": render_decision.get("profile_blend"),
            "mastering_intent": render_decision.get("mastering_intent"),
            "reference_v7": render_decision.get("reference_v7"),
            "detailed_genre_decision": render_decision.get("detailed_genre_decision"),
            "frequency_band_centrifuge_summary": (before.get("frequency_band_centrifuge") or {}).get("global_summary", {}) if isinstance(before, dict) and isinstance(before.get("frequency_band_centrifuge"), dict) else {},
        },
    }
    _debug("engine_done_pre_limiter_prepare", stage_gain_db=round(stage_gain_db, 3), character_scale=round(character_scale, 3))
    return np.nan_to_num(staged, nan=0.0, posinf=0.0, neginf=0.0), render_decision, prepare_report


def render_pre_limiter_from_prepared_stage(
    staged: np.ndarray,
    sr: int,
    decision: dict[str, Any],
    prepare_report: dict[str, Any],
    mode: str = "Auto Commercial Master",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Stage A-2b: render the pre-limiter master from an A-2a checkpoint."""
    runtime_intensity_lock = _refresh_runtime_mastering_policy(decision)
    _debug("engine_start_pre_limiter_render", sr=sr, shape=getattr(staged, "shape", None), dtype=str(getattr(staged, "dtype", "")), mode=mode, runtime_intensity_lock=runtime_intensity_lock)
    before = prepare_report.get("input_analysis", {}) or {}
    governor_report = prepare_report.get("loudness_push_governor", {}) or {}
    commercial_loudness_prep = prepare_report.get("commercial_loudness_prep", {}) or {}
    stage_gain_db = float(prepare_report.get("working_stage_gain_db", 0.0) or 0.0)
    character_scale = float(prepare_report.get("character_scale_initial", _boldness_scale(decision, mode)) or 1.0)

    true_pre_limiter_handoff_enabled = bool(_env_bool("BUSY_V6316_TRUE_PRE_LIMITER_HANDOFF", True))
    handoff_pre_limiter = None
    handoff_proc: dict[str, Any] = {}
    handoff_analysis: dict[str, Any] = {}
    handoff_render_settings = {"character_scale": character_scale, "loudness_offset_db": 0.0, "mb_amount_scale": 1.0}
    if true_pre_limiter_handoff_enabled:
        _debug("engine_start_v6316_true_pre_limiter_handoff_render")
        handoff_pre_limiter, handoff_proc = _render_once(
            staged, sr, decision, mode,
            character_scale=character_scale,
            loudness_offset_db=0.0,
            mb_amount_scale=1.0,
            final_loudness_stage=False,
        )
        handoff_analysis = analyze_audio_fast_qc(handoff_pre_limiter, sr, true_peak_oversample=_qc_oversample_factor())
        _debug("engine_done_v6316_true_pre_limiter_handoff_render", lufs=handoff_analysis.get("integrated_lufs"), true_peak=handoff_analysis.get("approx_true_peak_dbfs"), stages=handoff_proc.get("stages", []))

    _debug("engine_start_render_first_pass_limited_preview")
    first, first_proc = _render_once(staged, sr, decision, mode, character_scale=character_scale, loudness_offset_db=0.0, mb_amount_scale=1.0)
    _debug("engine_done_render_first_pass_limited_preview", stages=first_proc.get("stages", []), loudness_iteration=first_proc.get("loudness_iteration"))
    first_analysis = analyze_audio_fast_qc(first, sr, true_peak_oversample=_qc_oversample_factor())
    reasons = _safety_risk(before, first_analysis, decision)
    _debug("engine_done_first_pass_qc", reasons=reasons, lufs=first_analysis.get("integrated_lufs"), true_peak=first_analysis.get("approx_true_peak_dbfs"), crest=first_analysis.get("crest_factor_db"), correlation=first_analysis.get("correlation"))

    final = first
    final_analysis = first_analysis
    final_proc = first_proc
    rollback_used = False
    rollback_attempts: list[dict[str, Any]] = []

    v61_feedback_optimizer: dict[str, Any] = {"enabled": False, "active": False, "reason": "not_run"}
    if _env_bool("BUSY_V61_RENDER_MEASURED_FEEDBACK_OPTIMIZER", True):
        final, final_analysis, final_proc, reasons, v61_feedback_optimizer = _v61_render_measured_feedback_optimizer(
            staged,
            sr,
            decision,
            prepare_report,
            mode,
            character_scale,
            first,
            first_analysis,
            first_proc,
        )
        rollback_used = bool(v61_feedback_optimizer.get("rollback_count", 0))
        if v61_feedback_optimizer.get("active"):
            rollback_attempts.append({
                "type": "v61_render_measured_feedback_optimizer",
                "selected_pass": v61_feedback_optimizer.get("selected_pass"),
                "accepted_pass_count": v61_feedback_optimizer.get("accepted_pass_count"),
                "rollback_count": v61_feedback_optimizer.get("rollback_count"),
                "selected_score": v61_feedback_optimizer.get("selected_score"),
            })
        _debug(
            "engine_done_v61_feedback_optimizer",
            active=v61_feedback_optimizer.get("active"),
            selected_pass=v61_feedback_optimizer.get("selected_pass"),
            accepted_pass_count=v61_feedback_optimizer.get("accepted_pass_count"),
            rollback_count=v61_feedback_optimizer.get("rollback_count"),
            selected_score=v61_feedback_optimizer.get("selected_score"),
        )

    if reasons and not bool(v61_feedback_optimizer.get("selected_pass")):
        rollback_used = True
        _debug("engine_start_rollback", initial_reasons=reasons)
        all_candidates = [
            {"character_scale": character_scale * 0.78, "loudness_offset_db": -0.35, "mb_amount_scale": 0.92},
            {"character_scale": character_scale * 0.62, "loudness_offset_db": -0.75, "mb_amount_scale": 0.80},
            {"character_scale": character_scale * 0.48, "loudness_offset_db": -1.10, "mb_amount_scale": 0.68},
        ]
        max_attempts = _effective_rollback_max_attempts(staged, sr, decision)
        candidates = all_candidates[:max_attempts]
        _debug("engine_rollback_plan", rollback_max_attempts=max_attempts, candidate_count=len(candidates))
        best = (len(reasons), final, final_analysis, final_proc, reasons)
        for idx, c in enumerate(candidates, start=1):
            _debug("engine_start_rollback_attempt", attempt=idx, settings=c)
            cand, proc = _render_once(staged, sr, decision, mode, **c)
            cand_analysis = analyze_audio_fast_qc(cand, sr, true_peak_oversample=_qc_oversample_factor())
            cand_reasons = _safety_risk(before, cand_analysis, decision)
            rollback_attempts.append({"settings": c, "warnings": cand_reasons})
            _debug("engine_done_rollback_attempt", attempt=idx, warnings=cand_reasons, lufs=cand_analysis.get("integrated_lufs"), true_peak=cand_analysis.get("approx_true_peak_dbfs"), crest=cand_analysis.get("crest_factor_db"))
            score = len(cand_reasons)
            if score < best[0] or (score == best[0] and cand_analysis.get("crest_factor_db", 0) > best[2].get("crest_factor_db", 0)):
                best = (score, cand, cand_analysis, proc, cand_reasons)
                handoff_render_settings = dict(c)
            else:
                cand = None
            gc.collect()
        _, final, final_analysis, final_proc, reasons = best
        _debug("engine_done_rollback", remaining_reasons=reasons, selected_lufs=final_analysis.get("integrated_lufs"), selected_true_peak=final_analysis.get("approx_true_peak_dbfs"), selected_crest=final_analysis.get("crest_factor_db"))

    if true_pre_limiter_handoff_enabled and isinstance(handoff_render_settings, dict):
        try:
            _initial = {"character_scale": character_scale, "loudness_offset_db": 0.0, "mb_amount_scale": 1.0}
            _changed = any(abs(float(handoff_render_settings.get(k, _initial[k])) - float(_initial[k])) > 1e-6 for k in _initial)
        except Exception:
            _changed = False
        if _changed:
            try:
                _debug("engine_start_v6316_true_pre_limiter_replay_selected_settings", settings=handoff_render_settings)
                handoff_pre_limiter, handoff_proc = _render_once(
                    staged, sr, decision, mode,
                    character_scale=float(handoff_render_settings.get("character_scale", character_scale)),
                    loudness_offset_db=float(handoff_render_settings.get("loudness_offset_db", 0.0)),
                    mb_amount_scale=float(handoff_render_settings.get("mb_amount_scale", 1.0)),
                    final_loudness_stage=False,
                )
                handoff_analysis = analyze_audio_fast_qc(handoff_pre_limiter, sr, true_peak_oversample=_qc_oversample_factor())
                _debug("engine_done_v6316_true_pre_limiter_replay_selected_settings", lufs=handoff_analysis.get("integrated_lufs"), true_peak=handoff_analysis.get("approx_true_peak_dbfs"))
            except Exception as exc:
                _debug("engine_error_v6316_true_pre_limiter_replay_selected_settings", error=str(exc)[:240])

    targets = decision.get("targets", {}) if isinstance(decision, dict) else {}
    crest_floor = float(targets.get("crest_floor_db", 5.0))
    if final_analysis.get("crest_factor_db", 99) < crest_floor and "true_peak_too_hot" not in reasons:
        _debug("engine_start_micro_transient_restore", crest=final_analysis.get("crest_factor_db"), crest_floor=crest_floor)
        restored = transient_micro_expander(final, sr, amount=0.06)
        restored = lookahead_limiter(restored, sr, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=1.0, release_ms=140)
        restored_analysis = analyze_audio_fast_qc(restored, sr, true_peak_oversample=_qc_oversample_factor())
        restored_reasons = _safety_risk(before, restored_analysis, decision)
        if len(restored_reasons) <= len(reasons) and restored_analysis.get("crest_factor_db", 0) >= final_analysis.get("crest_factor_db", 0):
            final, final_analysis, reasons = restored, restored_analysis, restored_reasons
            final_proc.setdefault("stages", []).append("micro_transient_restore")
            _debug("engine_done_micro_transient_restore", reasons=reasons, crest=final_analysis.get("crest_factor_db"), true_peak=final_analysis.get("approx_true_peak_dbfs"))

    returned_pre_limiter = final
    returned_pre_limiter_analysis = final_analysis
    returned_processing_report = final_proc
    v6316_reconnect = {
        "schema_version": "busy_v6316_mastering_chain_reconnect_true_pre_limiter_handoff",
        "enabled": bool(true_pre_limiter_handoff_enabled),
        "active": False,
        "returned_checkpoint": "legacy_limited_preview",
        "limited_preview_lufs": final_analysis.get("integrated_lufs") if isinstance(final_analysis, dict) else None,
        "policy": "Pre-limiter output must be the real post-EQ/dynamics/polish checkpoint before loudness_iteration/limiter. The limited first pass remains only as preview/QC so finalize_limiter_master can run commercial finish, v60 DML, final limiter, dither, export lock, and QC in order.",
    }
    if true_pre_limiter_handoff_enabled and handoff_pre_limiter is not None and isinstance(handoff_analysis, dict) and handoff_analysis:
        returned_pre_limiter = handoff_pre_limiter
        returned_pre_limiter_analysis = handoff_analysis
        returned_processing_report = handoff_proc if isinstance(handoff_proc, dict) and handoff_proc else final_proc
        v6316_reconnect.update({
            "active": True,
            "returned_checkpoint": "true_pre_limiter_before_loudness_iteration",
            "returned_lufs": handoff_analysis.get("integrated_lufs"),
            "returned_true_peak": handoff_analysis.get("approx_true_peak_dbfs"),
            "limited_preview_lufs": final_analysis.get("integrated_lufs"),
            "limited_preview_true_peak": final_analysis.get("approx_true_peak_dbfs"),
            "selected_handoff_render_settings": handoff_render_settings,
        })

    report = {
        "engine_version": "busy_auto_mastering_v8_5_3_63_1_6_2_1_single_stage_v61_hotfix_pre_limiter",
        "stage": "pre_limiter",
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": prepare_report.get("original_input_analysis_before_pre_clean", {}),
        "residue_pre_clean_stage": prepare_report.get("residue_pre_clean_stage", {}),
        "pre_limiter_analysis_fast": returned_pre_limiter_analysis,
        "working_stage_gain_db": round(stage_gain_db, 3),
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
        "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        "spatial_fx_mastering": prepare_report.get("spatial_fx_mastering", decision.get("spatial_fx_mastering_plan", {})),
        "rollback_used": rollback_used,
        "rollback_attempts": rollback_attempts,
        "remaining_safety_warnings_pre_limiter": reasons,
        "risk_metrics_pre_limiter": _risk_metrics(before, returned_pre_limiter_analysis),
        "character_scale_initial": round(character_scale, 3),
        "genre_chain": decision.get("genre_chain", {}),
        "processing_report": returned_processing_report,
        "limited_preview_analysis_fast": final_analysis,
        "limited_preview_processing_report": final_proc,
        "v6316_mastering_chain_reconnect": v6316_reconnect,
        "axis_ownership_ledger": {
            "v58": ((returned_processing_report or {}).get("v58_spectral_conditioning", {}) or {}).get("axis_ownership_ledger", {}) if isinstance(final_proc, dict) else {},
            "v59": ((returned_processing_report or {}).get("v59_lf_crest_source_conditioning", {}) or {}).get("axis_ownership_ledger", {}) if isinstance(final_proc, dict) else {},
            "policy": "Top-level debug mirror; authoritative ledgers remain inside processing_report.v58/v59.",
        },
        "v61_render_measured_feedback_optimizer": v61_feedback_optimizer,
        "v62_8_mastering_blocker_preparation": decision.get("v62_8_mastering_blocker_preparation", {}) if isinstance(decision, dict) else {},
        "decision_summary": {
            "detected_style": decision.get("detected_style") or decision.get("genre_id"),
            "selected_profile": decision.get("selected_profile"),
            "style_confidence": decision.get("style_confidence"),
            "boldness_level": decision.get("boldness_level", "auto"),
            "profile_blend": decision.get("profile_blend"),
            "mastering_intent": decision.get("mastering_intent"),
            "reference_v7": decision.get("reference_v7"),
            "detailed_genre_decision": decision.get("detailed_genre_decision"),
            "frequency_band_centrifuge_summary": (before.get("frequency_band_centrifuge") or {}).get("global_summary", {}) if isinstance(before, dict) and isinstance(before.get("frequency_band_centrifuge"), dict) else {},
        },
    }
    _debug("engine_done_pre_limiter_chain", lufs=returned_pre_limiter_analysis.get("integrated_lufs"), true_peak=returned_pre_limiter_analysis.get("approx_true_peak_dbfs"), crest=returned_pre_limiter_analysis.get("crest_factor_db"), returned_checkpoint=v6316_reconnect.get("returned_checkpoint"))
    return np.nan_to_num(returned_pre_limiter, nan=0.0, posinf=0.0, neginf=0.0), report


def render_pre_limiter_master(
    y: np.ndarray,
    sr: int,
    decision: dict[str, Any],
    mode: str = "Auto Commercial Master",
    pre_analysis: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render everything up to the final oversampled limiter.

    This wrapper preserves the old API while internally matching the A-2a/A-2b
    split used by the Streamlit/Supabase pipeline.
    """
    runtime_intensity_lock = _refresh_runtime_mastering_policy(decision)
    _debug("engine_start_pre_limiter_chain", sr=sr, shape=getattr(y, "shape", None), dtype=str(getattr(y, "dtype", "")), mode=mode, runtime_intensity_lock=runtime_intensity_lock)
    staged, render_decision, prepare_report = prepare_pre_limiter_stage(y, sr, decision, mode=mode, pre_analysis=pre_analysis)
    return render_pre_limiter_from_prepared_stage(staged, sr, render_decision, prepare_report, mode=mode)



def _master_score_for_optimizer(before: dict[str, Any], after: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[float, list[str]]:
    """Score a rendered candidate for safe automatic selection.

    The score intentionally rewards translation and controlled loudness, not just
    the loudest LUFS. It is used for lightweight multi-pass selection in the
    fresh limiter worker.
    """
    warnings = []
    try:
        warnings.extend(_safety_risk(before, after, decision))
    except Exception:
        pass
    try:
        warnings.extend(_stereo_qc_summary(after).get("warnings", []))
    except Exception:
        pass
    try:
        warnings.extend(_ab_delta_qc(before, after).get("warnings", []))
    except Exception:
        pass
    try:
        warnings.extend(_lra_guard_summary(before, after, decision, mode).get("warnings", []))
    except Exception:
        pass
    warnings = sorted(set(warnings))

    lufs = float(after.get("integrated_lufs", -99) or -99)
    crest = float(after.get("crest_factor_db", 0) or 0)
    tp = float(after.get("approx_true_peak_dbfs", -99) or -99)
    corr = float(after.get("correlation", 1.0) or 1.0)
    q = after.get("quality_indices", {}) if isinstance(after, dict) else {}
    harsh = float(q.get("harshness_index_db", 0) or 0)
    air = float(q.get("air_index_db", 0) or 0)
    ms = after.get("ms", {}) if isinstance(after, dict) else {}
    low_side = float(ms.get("side_ratio_lowband", 0) or 0)
    high_side = float(ms.get("side_ratio_8k_14k", 0) or 0)

    score = 100.0
    score -= 18.0 * len(warnings)
    score += max(-20.0, min(16.0, (lufs + 12.0) * 3.0))
    if crest < 7.0:
        score -= (7.0 - crest) * 8.0
    elif crest > 13.0:
        score -= min(8.0, (crest - 13.0) * 1.5)
    else:
        score += min(8.0, (crest - 7.0) * 1.2)
    if tp > FINAL_TRUE_PEAK_CEILING_DB + 0.03:
        score -= 20.0
    if corr < 0.08:
        score -= 18.0
    if low_side > 0.16:
        score -= (low_side - 0.16) * 90.0
    if high_side > 0.72 and corr < 0.25:
        score -= 8.0
    if harsh > 0.0:
        score -= min(8.0, harsh * 0.7)
    if air > 0.0 and high_side > 0.6:
        score -= min(8.0, air * 0.5)
    return float(round(score, 3)), warnings



# ---------------------------------------------------------------------------
# v8.5.3.61: Render-Measured Source Conditioning Feedback Optimizer.
# ---------------------------------------------------------------------------

def _v61_metric(obj: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    if not isinstance(obj, dict):
        return default
    for key in keys:
        try:
            val = obj.get(key)
            if val is not None:
                f = float(val)
                if np.isfinite(f):
                    return float(f)
        except Exception:
            pass
    if any("lra" in str(k).lower() for k in keys):
        try:
            val = _analysis_lra_lu(obj)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    return default


def _v61_ms_value(analysis: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    try:
        ms = analysis.get("ms", {}) if isinstance(analysis, dict) and isinstance(analysis.get("ms", {}), dict) else {}
        val = float(ms.get(key, default) or default)
        return float(val) if np.isfinite(val) else float(default)
    except Exception:
        return float(default)


def _v61_q_value(analysis: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    try:
        q = analysis.get("quality_indices", {}) if isinstance(analysis, dict) and isinstance(analysis.get("quality_indices", {}), dict) else {}
        val = float(q.get(key, default) or default)
        return float(val) if np.isfinite(val) else float(default)
    except Exception:
        return float(default)


def _v61_blockers(before: dict[str, Any], after: dict[str, Any], decision: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    """Extract measured blockers from an actual rendered pass.

    This is deliberately conservative: it does not invent corrections from a
    target LUFS miss.  Only measured damage/blocker axes create actions.
    """
    blockers: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        warnings.extend(_safety_risk(before, after, decision))
    except Exception:
        pass
    try:
        warnings.extend(_stereo_qc_summary(after).get("warnings", []))
    except Exception:
        pass
    try:
        warnings.extend(_ab_delta_qc(before, after).get("warnings", []))
    except Exception:
        pass
    try:
        warnings.extend(_lra_guard_summary(before, after, decision, mode).get("warnings", []))
    except Exception:
        pass
    warnings = sorted(set([str(w) for w in warnings if w]))

    lra = _v61_metric(after, "lra_lu", "loudness_range_lu", default=None)
    lra_src = _v61_metric(before, "lra_lu", "loudness_range_lu", default=None)
    crest = _v61_metric(after, "crest_factor_db", default=None)
    crest_src = _v61_metric(before, "crest_factor_db", default=None)
    tp = _v61_metric(after, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
    corr = _v61_metric(after, "correlation", default=1.0)
    low_side = _v61_ms_value(after, "side_ratio_lowband", 0.0)
    high_side = _v61_ms_value(after, "side_ratio_8k_14k", 0.0)
    harsh = _v61_q_value(after, "harshness_index_db", 0.0)
    air = _v61_q_value(after, "air_index_db", 0.0)
    lowmid_q = _v61_q_value(after, "mud_index_db", 0.0)
    if lowmid_q <= 0.0:
        lowmid_q = _v61_q_value(after, "low_mid_congestion_db", 0.0)

    def add(axis: str, severity: float, reason: str, indicators: dict[str, Any] | None = None) -> None:
        sev = float(np.clip(float(severity), 0.0, 1.0))
        if sev <= 0.0:
            return
        blockers.append({
            "axis": axis,
            "severity": round(sev, 4),
            "reason": reason,
            "indicators": indicators or {},
        })

    try:
        if tp is not None and float(tp) > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.05:
            add("true_peak_pressure", min(1.0, (float(tp) - float(FINAL_TRUE_PEAK_CEILING_DB)) / 0.75), "true_peak_above_locked_ceiling", {"true_peak_dbtp": round(float(tp), 3)})
        elif "true_peak_too_hot" in warnings:
            add("true_peak_pressure", 0.65, "safety_warning_true_peak_too_hot", {"true_peak_dbtp": round(float(tp or -99.0), 3)})
    except Exception:
        pass
    try:
        if lra is not None:
            lra_floor = _env_float("BUSY_V61_LRA_HARD_FLOOR_LU", 2.5, 1.0, 6.0)
            source_already_flat = bool(lra_src is not None and float(lra_src) < lra_floor)
            if float(lra) < lra_floor and not source_already_flat:
                add("lra_collapse", 1.0, "lra_hard_floor", {"lra_lu": round(float(lra), 3), "lra_floor_lu": round(float(lra_floor), 3)})
            elif lra_src is not None and float(lra_src) > 0.75 and float(lra) < float(lra_src) * 0.72:
                add("lra_collapse", min(1.0, (float(lra_src) * 0.72 - float(lra)) / max(0.25, float(lra_src) * 0.35)), "lra_loss_vs_source", {"source_lra_lu": round(float(lra_src), 3), "candidate_lra_lu": round(float(lra), 3)})
            elif ("lra_too_flat" in warnings or "lra_reduced_too_much" in warnings) and not source_already_flat:
                add("lra_collapse", 0.74, "lra_guard_warning", {"lra_lu": round(float(lra), 3)})
    except Exception:
        pass
    try:
        if crest is not None:
            if crest_src is not None and float(crest_src) > 1.0 and float(crest) < float(crest_src) - 0.85:
                add("crest_transient_loss", min(1.0, (float(crest_src) - float(crest)) / 3.0), "crest_loss_vs_source", {"source_crest_db": round(float(crest_src), 3), "candidate_crest_db": round(float(crest), 3)})
            elif float(crest) < _env_float("BUSY_V61_CREST_SOFT_FLOOR_DB", 6.5, 3.0, 12.0) or "transient_loss_delta" in warnings:
                add("crest_transient_loss", 0.58, "crest_or_transient_floor", {"crest_factor_db": round(float(crest), 3)})
    except Exception:
        pass
    try:
        if low_side > 0.12:
            add("side_low_instability", min(1.0, (low_side - 0.12) / 0.20), "low_band_side_ratio_high", {"side_ratio_lowband": round(float(low_side), 4)})
        if high_side > 0.62 or (high_side > 0.50 and corr is not None and corr < 0.25):
            add("side_high_spike", min(1.0, max((high_side - 0.50) / 0.35, 0.35 if corr is not None and corr < 0.20 else 0.0)), "side_high_spike_or_correlation_risk", {"side_ratio_8k_14k": round(float(high_side), 4), "correlation": round(float(corr or 0.0), 4)})
        if corr is not None and float(corr) < 0.08:
            add("correlation_risk", min(1.0, (0.08 - float(corr)) / 0.24 + 0.45), "phase_correlation_low", {"correlation": round(float(corr), 4)})
    except Exception:
        pass
    try:
        if lowmid_q > 0.35 or "low_mid_mud_delta" in warnings:
            add("low_mid_bottleneck", min(1.0, max(lowmid_q / 3.0, 0.42)), "low_mid_congestion_indicator", {"low_mid_indicator": round(float(lowmid_q), 4)})
        if harsh > 0.0 or "harshness_delta_excess" in warnings:
            add("harsh_upper_mid", min(1.0, max(harsh / 5.0, 0.35)), "harshness_indicator", {"harshness_index_db": round(float(harsh), 4)})
        if air > 0.0 and high_side > 0.45:
            add("hf_fizz_side_hash", min(1.0, max(air / 5.0, 0.30)), "air_hash_with_side_energy", {"air_index_db": round(float(air), 4), "side_ratio_8k_14k": round(float(high_side), 4)})
    except Exception:
        pass
    for w in warnings:
        if w in {"limiter_overwork", "clipping_distortion_risk"}:
            add("limiter_overwork", 0.70, f"safety_warning_{w}", {})
    # Deduplicate by axis, keeping the highest severity.
    best: dict[str, dict[str, Any]] = {}
    for b in blockers:
        axis = str(b.get("axis") or "")
        if not axis:
            continue
        if axis not in best or float(b.get("severity", 0.0) or 0.0) > float(best[axis].get("severity", 0.0) or 0.0):
            best[axis] = b
    return sorted(best.values(), key=lambda x: float(x.get("severity", 0.0) or 0.0), reverse=True)


def _v61_append_move(plan: dict[str, Any], bucket: str, move: dict[str, Any], cap_abs_db: float) -> None:
    moves = plan.get(bucket)
    if not isinstance(moves, list):
        moves = []
    m = dict(move)
    try:
        gain = float(m.get("gain_db", 0.0) or 0.0)
        # Feedback optimizer is corrective-only for v61. Never convert a measured
        # blocker into a boost.  Small positive moves are clamped to zero.
        if gain > 0.0:
            gain = 0.0
        gain = float(np.clip(gain, -abs(float(cap_abs_db)), 0.0))
        m["gain_db"] = round(gain, 3)
    except Exception:
        pass
    m.setdefault("source", "v61_render_measured_feedback_optimizer")
    moves.append(m)
    plan[bucket] = moves


def _v61_feedback_decision(base_decision: dict[str, Any], blockers: list[dict[str, Any]], pass_idx: int, trust_scale: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a decision copy with measured-blocker corrections for one rerender.

    The update policy is deliberately small, dynamic-first, and no-boost.  The
    loop proves each update by rerendering from the same staged source.
    """
    d = copy.deepcopy(base_decision if isinstance(base_decision, dict) else {})
    trust = float(np.clip(float(trust_scale), 0.15, 1.0))
    pass_gain = float(np.clip(0.70 + 0.15 * max(0, pass_idx - 1), 0.70, 1.0))
    actions: list[dict[str, Any]] = []
    spectral = d.get("v58_spectral_conditioning_plan") if isinstance(d.get("v58_spectral_conditioning_plan"), dict) else {}
    spectral = copy.deepcopy(spectral) if spectral else {"schema_version": "busy_v58_spectral_conditioning_v8_5_3_58_1", "enabled": True, "active": False, "render_affecting": False, "static_moves": [], "dynamic_moves": [], "reasons": []}
    spectral.setdefault("static_moves", [])
    spectral.setdefault("dynamic_moves", [])
    spectral.setdefault("reasons", [])
    lf = d.get("v59_lf_crest_source_conditioning_plan") if isinstance(d.get("v59_lf_crest_source_conditioning_plan"), dict) else {}
    lf = copy.deepcopy(lf) if lf else {"schema_version": "busy_v59_lf_crest_source_conditioning_v8_5_3_59_2", "enabled": True, "active": False, "render_affecting": False, "static_moves": [], "dynamic_moves": [], "time_domain_actions": [], "stereo_side_actions": [], "reasons": []}
    for key in ("static_moves", "dynamic_moves", "time_domain_actions", "stereo_side_actions", "reasons", "actions"):
        lf.setdefault(key, [])
    v59_2 = lf.get("v59_2_program_dependent_dynamics_stereo_side") if isinstance(lf.get("v59_2_program_dependent_dynamics_stereo_side"), dict) else {}
    v59_2 = copy.deepcopy(v59_2) if v59_2 else {"schema_version": "busy_v59_2_program_dependent_dynamics_stereo_side", "enabled": True, "active": False, "render_affecting": False, "actions": [], "stereo_side_actions": [], "reasons": [], "program_dynamics_controls": {}}
    controls = v59_2.get("program_dynamics_controls") if isinstance(v59_2.get("program_dynamics_controls"), dict) else {}
    controls = copy.deepcopy(controls)
    loudness_offset_db = 0.0
    mb_amount_scale = 1.0
    character_scale_multiplier = 1.0
    blocker_axes = [str(b.get("axis") or "") for b in blockers]

    def sev(axis: str) -> float:
        vals = [float(b.get("severity", 0.0) or 0.0) for b in blockers if b.get("axis") == axis]
        return max(vals) if vals else 0.0

    if "low_mid_bottleneck" in blocker_axes:
        s = sev("low_mid_bottleneck") * trust * pass_gain
        _v61_append_move(spectral, "dynamic_moves", {"type": "bell", "freq": 360, "gain_db": -0.18 - 0.42 * s, "q": 0.95, "target": "mid", "source": "v61_low_mid_bottleneck_dynamic_decongestion"}, 0.95)
        _v61_append_move(spectral, "dynamic_moves", {"type": "bell", "freq": 520, "gain_db": -0.08 - 0.24 * s, "q": 1.05, "target": "mid", "source": "v61_masking_like_lowmid_release"}, 0.65)
        loudness_offset_db -= min(0.28, 0.10 + 0.20 * s)
        actions.append({"axis": "low_mid_bottleneck", "update": "small_dynamic_low_mid_decongestion", "strength": round(s, 4)})
    if "harsh_upper_mid" in blocker_axes:
        s = sev("harsh_upper_mid") * trust * pass_gain
        _v61_append_move(spectral, "dynamic_moves", {"type": "bell", "freq": 3600, "gain_db": -0.10 - 0.38 * s, "q": 1.25, "target": "stereo", "source": "v61_harsh_upper_mid_dynamic_guard"}, 0.85)
        actions.append({"axis": "harsh_upper_mid", "update": "small_dynamic_presence_guard", "strength": round(s, 4)})
    if "hf_fizz_side_hash" in blocker_axes:
        s = sev("hf_fizz_side_hash") * trust * pass_gain
        _v61_append_move(spectral, "dynamic_moves", {"type": "high_shelf", "freq": 10500, "gain_db": -0.08 - 0.30 * s, "q": 0.75, "target": "side", "source": "v61_hf_fizz_side_hash_guard"}, 0.65)
        actions.append({"axis": "hf_fizz_side_hash", "update": "side_high_dynamic_hash_guard", "strength": round(s, 4)})
    if "side_low_instability" in blocker_axes:
        s = sev("side_low_instability") * trust * pass_gain
        _v61_append_move(lf, "static_moves", {"type": "bell", "freq": 135, "gain_db": -0.08 - 0.22 * s, "q": 0.80, "target": "side", "source": "v61_side_low_stability_trim"}, 0.55)
        _v61_append_move(lf, "dynamic_moves", {"type": "bell", "freq": 92, "gain_db": -0.12 - 0.32 * s, "q": 0.88, "target": "side", "source": "v61_lf_side_dynamic_restraint"}, 0.75)
        actions.append({"axis": "side_low_instability", "update": "bounded_lf_side_restraint", "strength": round(s, 4)})
    if "side_high_spike" in blocker_axes or "correlation_risk" in blocker_axes:
        s = max(sev("side_high_spike"), sev("correlation_risk")) * trust * pass_gain
        side_action = {"module": "M11_SBST", "action": "side_band_spike_tamer", "strength": round(float(np.clip(0.24 + 0.38 * s, 0.18, 0.68)), 4), "max_gr_db": round(float(np.clip(0.40 + 0.70 * s, 0.35, 1.10)), 3), "source": "v61_feedback_side_band_spike_tamer"}
        lf.setdefault("stereo_side_actions", []).append(side_action)
        v59_2.setdefault("stereo_side_actions", []).append(side_action)
        v59_2.setdefault("actions", []).append(side_action)
        v59_2.setdefault("reasons", []).append("v61_measured_side_high_or_correlation_feedback")
        actions.append({"axis": "side_high_spike_or_correlation", "update": "bounded_side_band_spike_tamer", "strength": round(s, 4)})
    if "crest_transient_loss" in blocker_axes or "lra_collapse" in blocker_axes or "true_peak_pressure" in blocker_axes or "limiter_overwork" in blocker_axes:
        damage_s = max(sev("crest_transient_loss"), sev("lra_collapse"), sev("true_peak_pressure"), sev("limiter_overwork")) * trust * pass_gain
        loudness_offset_db -= min(0.95, 0.28 + 0.62 * damage_s)
        mb_amount_scale = min(mb_amount_scale, float(np.clip(1.0 - 0.10 * damage_s, 0.78, 1.0)))
        character_scale_multiplier = min(character_scale_multiplier, float(np.clip(1.0 - 0.08 * damage_s, 0.82, 1.0)))
        controls["multiband_amount_scale"] = round(float(np.clip(float(controls.get("multiband_amount_scale", 1.0) or 1.0) * (1.0 - 0.08 * damage_s), 0.74, 1.04)), 4)
        controls["dynamic_eq_intensity_scale"] = round(float(np.clip(float(controls.get("dynamic_eq_intensity_scale", 1.0) or 1.0) * (1.0 - 0.06 * damage_s), 0.78, 1.04)), 4)
        controls["upward_density_scale"] = round(float(np.clip(float(controls.get("upward_density_scale", 1.0) or 1.0) * (1.0 - 0.18 * damage_s), 0.42, 1.02)), 4)
        if "true_peak_pressure" in blocker_axes or "limiter_overwork" in blocker_axes:
            lf.setdefault("time_domain_actions", []).append({"module": "M3_M4", "action": "rogue_spike_outlier_tamer", "percentile": 99.90, "max_gr_db": round(float(np.clip(0.32 + 0.42 * damage_s, 0.25, 0.85)), 3), "source": "v61_feedback_rogue_peak_hygiene"})
        actions.append({"axis": "lra_crest_tp_or_limiter_damage", "update": "reduce_push_and_dynamics_density", "strength": round(damage_s, 4)})
    if "bass_headroom_cost" in blocker_axes:
        s = sev("bass_headroom_cost") * trust * pass_gain
        _v61_append_move(lf, "dynamic_moves", {"type": "bell", "freq": 74, "gain_db": -0.14 - 0.36 * s, "q": 0.78, "target": "stereo", "source": "v61_bass_headroom_dynamic_control"}, 0.80)
        loudness_offset_db -= min(0.35, 0.12 + 0.20 * s)
        actions.append({"axis": "bass_headroom_cost", "update": "lf_dynamic_headroom_control", "strength": round(s, 4)})

    if actions:
        spectral["enabled"] = True
        spectral["active"] = True
        spectral["render_affecting"] = True
        spectral.setdefault("reasons", []).append("v61_measured_feedback_overlay")
        lf["enabled"] = True
        lf["active"] = True
        lf["render_affecting"] = True
        lf.setdefault("reasons", []).append("v61_measured_feedback_overlay")
        v59_2["enabled"] = True
        v59_2["active"] = bool(v59_2.get("actions") or any(abs(float(controls.get(k, 1.0) or 1.0) - 1.0) > 1e-4 for k in ("multiband_amount_scale", "dynamic_eq_intensity_scale", "upward_density_scale")))
        v59_2["render_affecting"] = bool(v59_2.get("stereo_side_actions") or v59_2.get("active"))
    controls.setdefault("policy", "v61 may only downscale density/push when measured LRA/crest/TP blockers require it; no target-LUFS command.")
    v59_2["program_dynamics_controls"] = controls
    lf["v59_2_program_dependent_dynamics_stereo_side"] = v59_2
    lf["stereo_side_action_count"] = len(lf.get("stereo_side_actions", []) or [])
    lf["dynamic_move_count"] = len(lf.get("dynamic_moves", []) or [])
    lf["static_move_count"] = len(lf.get("static_moves", []) or [])
    lf["time_domain_action_count"] = len(lf.get("time_domain_actions", []) or [])
    spectral["dynamic_move_count"] = len(spectral.get("dynamic_moves", []) or [])
    spectral["static_move_count"] = len(spectral.get("static_moves", []) or [])
    d["v58_spectral_conditioning_plan"] = spectral
    d["v59_lf_crest_source_conditioning_plan"] = lf
    flags = list(d.get("processing_flags") or []) if isinstance(d.get("processing_flags"), list) else []
    if "v61_render_measured_feedback_optimizer" not in flags:
        flags.append("v61_render_measured_feedback_optimizer")
    d["processing_flags"] = flags
    update = {
        "actions": actions,
        "loudness_offset_db": round(float(loudness_offset_db), 4),
        "mb_amount_scale": round(float(mb_amount_scale), 4),
        "character_scale_multiplier": round(float(character_scale_multiplier), 4),
        "trust_scale": round(float(trust), 4),
        "policy": "blocker-triggered, bounded, dynamic-first, no-boost corrections; every update must survive rerender scoring or be rolled back.",
    }
    return d, update


def _v61_score_pass(before: dict[str, Any], analysis: dict[str, Any], decision: dict[str, Any], mode: str, applied_action_count: int = 0) -> tuple[float, dict[str, Any], list[str]]:
    base_score, warnings = _master_score_for_optimizer(before, analysis, decision, mode)
    lra = _v61_metric(analysis, "lra_lu", "loudness_range_lu", default=None)
    lra_src = _v61_metric(before, "lra_lu", "loudness_range_lu", default=None)
    crest = _v61_metric(analysis, "crest_factor_db", default=None)
    crest_src = _v61_metric(before, "crest_factor_db", default=None)
    tp = _v61_metric(analysis, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
    low_side = _v61_ms_value(analysis, "side_ratio_lowband", 0.0)
    high_side = _v61_ms_value(analysis, "side_ratio_8k_14k", 0.0)
    corr = _v61_metric(analysis, "correlation", default=1.0)
    components: dict[str, Any] = {"legacy_score": round(float(base_score), 4)}
    score = float(base_score)
    if lra is not None:
        lra_floor = _env_float("BUSY_V61_LRA_HARD_FLOOR_LU", 2.5, 1.0, 6.0)
        source_already_flat = bool(lra_src is not None and float(lra_src) < lra_floor)
        if float(lra) < lra_floor and not source_already_flat:
            score -= 80.0
            components["hard_lra_floor_penalty"] = 80.0
        elif lra_src is not None and float(lra_src) > 0.75 and float(lra) < float(lra_src) * 0.72:
            pen = min(45.0, (float(lra_src) * 0.72 - float(lra)) * 16.0)
            score -= pen
            components["lra_preservation_penalty"] = round(float(pen), 4)
        elif source_already_flat and lra_src is not None and float(lra) < float(lra_src) - 0.18:
            pen = min(24.0, (float(lra_src) - float(lra)) * 18.0)
            score -= pen
            components["additional_lra_damage_penalty"] = round(float(pen), 4)
    if crest is not None and crest_src is not None and float(crest_src) > 1.0 and float(crest) < float(crest_src) - 1.10:
        pen = min(35.0, (float(crest_src) - float(crest) - 1.10) * 9.0)
        score -= pen
        components["crest_preservation_penalty"] = round(float(pen), 4)
    if tp is not None and float(tp) > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.05:
        pen = min(70.0, 24.0 + (float(tp) - float(FINAL_TRUE_PEAK_CEILING_DB)) * 30.0)
        score -= pen
        components["true_peak_pressure_penalty"] = round(float(pen), 4)
    if low_side > 0.16:
        pen = min(20.0, (low_side - 0.16) * 90.0)
        score -= pen
        components["side_low_penalty"] = round(float(pen), 4)
    if high_side > 0.72 and corr is not None and float(corr) < 0.28:
        pen = min(18.0, (high_side - 0.72) * 25.0 + (0.28 - float(corr)) * 20.0)
        score -= pen
        components["side_high_correlation_penalty"] = round(float(pen), 4)
    if applied_action_count > 0:
        pen = min(8.0, float(applied_action_count) * _env_float("BUSY_V61_OVERPROCESSING_ACTION_PENALTY", 0.18, 0.0, 2.0))
        score -= pen
        components["overprocessing_penalty"] = round(float(pen), 4)
    components["score"] = round(float(score), 4)
    return round(float(score), 4), components, warnings


def _v61_hard_reject(before: dict[str, Any], analysis: dict[str, Any], warnings: list[str]) -> list[str]:
    reasons: list[str] = []
    tp = _v61_metric(analysis, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
    lra = _v61_metric(analysis, "lra_lu", "loudness_range_lu", default=None)
    lra_src = _v61_metric(before, "lra_lu", "loudness_range_lu", default=None)
    try:
        if tp is not None and float(tp) > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.30:
            reasons.append("hard_true_peak_pressure")
    except Exception:
        pass
    try:
        if lra is not None and float(lra) < 2.30 and not (lra_src is not None and float(lra_src) < 2.50):
            reasons.append("hard_lra_collapse")
    except Exception:
        pass
    if "phase_collapse" in warnings or "low_end_collapse" in warnings:
        reasons.append("hard_stereo_or_low_end_collapse")
    return sorted(set(reasons))




def _v61_feedback_status_reasons(
    status: str,
    cand_score: float,
    best_score: float,
    epsilon: float,
    hard_reject: list[str],
    cand_blockers: list[dict[str, Any]],
    cand_components: dict[str, Any],
    cand_warnings: list[str],
    update: dict[str, Any],
) -> list[str]:
    """Create compact human-readable pass-level accept/reject reasons."""
    reasons: list[str] = []
    try:
        delta = float(cand_score) - float(best_score)
    except Exception:
        delta = 0.0
    if status == "ACCEPTED":
        reasons.append(f"accepted_score_delta_{delta:.3f}_gte_epsilon_{float(epsilon):.3f}")
    elif hard_reject:
        reasons.extend([f"hard_gate:{str(r)}" for r in hard_reject])
    else:
        if delta <= 0.0:
            reasons.append(f"score_not_improved_delta_{delta:.3f}")
        elif delta < float(epsilon):
            reasons.append(f"score_delta_below_epsilon_{delta:.3f}_lt_{float(epsilon):.3f}")
        else:
            reasons.append("not_selected_by_unknown_soft_gate")
    blockers = [str(b.get("axis") or "") for b in cand_blockers if isinstance(b, dict) and b.get("axis")]
    if blockers:
        reasons.append("remaining_blockers:" + ",".join(blockers[:5]))
    for key in (
        "hard_lra_floor_penalty",
        "additional_lra_damage_penalty",
        "lra_preservation_penalty",
        "crest_preservation_penalty",
        "true_peak_pressure_penalty",
        "side_low_penalty",
        "side_high_correlation_penalty",
        "overprocessing_penalty",
    ):
        if key in cand_components:
            try:
                reasons.append(f"penalty:{key}={float(cand_components.get(key)):.3f}")
            except Exception:
                reasons.append(f"penalty:{key}")
    actions = update.get("actions") if isinstance(update, dict) else []
    if isinstance(actions, list) and actions:
        axes = [str(a.get("axis") or "") for a in actions if isinstance(a, dict) and a.get("axis")]
        if axes:
            reasons.append("attempted_actions:" + ",".join(axes[:5]))
    warning_keep = [str(w) for w in (cand_warnings or []) if str(w) in {"lra_too_flat", "lra_reduced_too_much", "true_peak_too_hot", "transient_loss_delta", "limiter_overwork", "clipping_distortion_risk", "phase_collapse", "low_end_collapse"}]
    if warning_keep:
        reasons.append("warnings:" + ",".join(sorted(set(warning_keep))[:5]))
    return reasons[:12]


# ---------------------------------------------------------------------------
# v8.5.3.62.8 - Mastering Blocker Preparation / Safe Push Feasibility
# ---------------------------------------------------------------------------
def _v62_8_num(value: Any, default: float | None = None) -> float | None:
    try:
        f = float(value)
        if np.isfinite(f):
            return float(f)
    except Exception:
        pass
    return default


def _v62_8_metric(d: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            v = _v62_8_num(d.get(k), None)
            if v is not None:
                return v
    loud = d.get("loudness") if isinstance(d.get("loudness"), dict) else {}
    for k in keys:
        if isinstance(loud, dict) and k in loud:
            v = _v62_8_num(loud.get(k), None)
            if v is not None:
                return v
    return default


def _v62_8_band_db(analysis: dict[str, Any] | None, *names: str, default: float = -90.0) -> float:
    """Best-effort spectral-band resolver tolerant of existing report schemas."""
    if not isinstance(analysis, dict):
        return float(default)
    candidates: list[Any] = []
    for n in names:
        candidates.append(analysis.get(n))
    for container_key in ("bands_db", "band_db", "frequency_bands_db", "frequency_bands", "spectrum_bands", "band_energy_db"):
        c = analysis.get(container_key)
        if isinstance(c, dict):
            for n in names:
                candidates.append(c.get(n))
    fbc = analysis.get("frequency_band_centrifuge") if isinstance(analysis.get("frequency_band_centrifuge"), dict) else {}
    for key in ("global_summary", "band_summary", "bands", "band_energy_db"):
        c = fbc.get(key) if isinstance(fbc, dict) else None
        if isinstance(c, dict):
            for n in names:
                candidates.append(c.get(n))
    for v in candidates:
        fv = _v62_8_num(v, None)
        if fv is not None:
            return float(fv)
    return float(default)


def _v62_8_safe_push_feasibility(before: dict[str, Any] | None, analysis: dict[str, Any] | None, decision: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    """Estimate commercial loudness pushability without commanding a push.

    This deliberately separates feasibility from processing.  It tells later stages why
    a track is not safely pushable; it never overrides v57/v60 damage gates or TP locks.
    """
    a = analysis if isinstance(analysis, dict) else (before if isinstance(before, dict) else {})
    b = before if isinstance(before, dict) else {}
    targets = (decision or {}).get("targets", {}) if isinstance(decision, dict) and isinstance((decision or {}).get("targets"), dict) else {}
    current_lufs = _v62_8_metric(a, "integrated_lufs", default=_v62_8_metric(b, "integrated_lufs", default=-99.0))
    target_lufs = _v62_8_num(targets.get("target_lufs"), None)
    if target_lufs is None:
        # Use the existing commercial finish profile source where available, but stay safe.
        try:
            target_lufs = float(_commercial_loudness_finish_profile(decision or {}, mode).get("target_lufs", -8.0) or -8.0)
        except Exception:
            target_lufs = -8.0
    tp = _v62_8_metric(a, "approx_true_peak_dbfs", "true_peak_dbtp", default=None)
    crest = _v62_8_metric(a, "crest_factor_db", default=None)
    lra = _v62_8_metric(a, "lra_lu", "loudness_range_lu", default=None)
    side_low = _v61_ms_value(a, "side_ratio_lowband", 0.0) if isinstance(a, dict) else 0.0
    # Proxy low-mid pressure.  Prefer available ratios; otherwise approximate from band dB.
    low_mid_ratio = _v62_8_metric(a, "low_mid_ratio", "low_mid_bottleneck_ratio", default=None)
    if low_mid_ratio is None:
        low_mid_db = _v62_8_band_db(a, "low_mid", "lowmid", "band_120_500", "120_500", default=-90.0)
        mid_db = _v62_8_band_db(a, "mid", "mid_body", "band_500_2000", "500_2000", default=-90.0)
        try:
            low_mid_ratio = float(1.0 / (1.0 + np.exp(-(float(low_mid_db) - float(mid_db) - 1.5) / 3.0)))
        except Exception:
            low_mid_ratio = 0.0
    crest_margin = max(0.0, float(crest or 0.0) - _env_float("BUSY_V62_8_CREST_SAFE_FLOOR_DB", 8.0, 4.0, 14.0)) if crest is not None else 0.0
    lra_margin = max(0.0, float(lra or 0.0) - _env_float("BUSY_V62_8_LRA_SAFE_FLOOR_LU", 4.0, 1.0, 10.0)) if lra is not None else 0.0
    desired_push = max(0.0, float(target_lufs or -8.0) - float(current_lufs or -99.0))
    tp_margin = max(0.0, float(FINAL_TRUE_PEAK_CEILING_DB) - float(tp or -99.0)) if tp is not None else desired_push
    low_mid_cost = 0.0
    if float(low_mid_ratio or 0.0) >= _env_float("BUSY_V62_8_LOWMID_RATIO_COST_THRESHOLD", 0.40, 0.1, 0.9):
        low_mid_cost = _env_float("BUSY_V62_8_LOWMID_BUDGET_COST_DB", 1.5, 0.0, 4.0)
    side_low_cost = min(1.0, max(0.0, float(side_low) - 0.18) * 4.0)
    raw_budget = min(desired_push, crest_margin, lra_margin, tp_margin)
    safe_budget = max(0.0, raw_budget - low_mid_cost - side_low_cost)
    reasons: list[str] = []
    if crest is not None and crest_margin <= 0.05:
        reasons.append("crest_margin_exhausted")
    if lra is not None and lra_margin <= 0.05:
        reasons.append("lra_margin_exhausted")
    if tp is not None and tp_margin <= 0.05:
        reasons.append("true_peak_margin_exhausted")
    if low_mid_cost > 0:
        reasons.append("low_mid_bottleneck_cost")
    if side_low_cost > 0:
        reasons.append("side_low_headroom_cost")
    if not reasons:
        reasons.append("bounded_safe_push_possible")
    return {
        "schema_version": "busy_v62_8_commercial_loudness_feasibility_v8_5_3_62_8",
        "current_lufs": round(float(current_lufs or 0.0), 3),
        "target_lufs": round(float(target_lufs or 0.0), 3),
        "desired_push_db": round(float(desired_push), 3),
        "safe_push_budget_db": round(float(safe_budget), 3),
        "useful_safe_push_available": bool(safe_budget >= _env_float("BUSY_V62_8_MIN_USEFUL_PUSH_DB", 0.35, 0.0, 3.0)),
        "budget_by_axis_db": {
            "tp_margin": round(float(tp_margin), 3),
            "crest_margin": round(float(crest_margin), 3),
            "lra_margin": round(float(lra_margin), 3),
            "low_mid_cost": round(float(low_mid_cost), 3),
            "side_low_cost": round(float(side_low_cost), 3),
        },
        "low_mid_ratio_proxy": round(float(low_mid_ratio or 0.0), 4),
        "side_low_ratio": round(float(side_low), 4),
        "decision": "allow_bounded_safe_push" if safe_budget >= _env_float("BUSY_V62_8_MIN_USEFUL_PUSH_DB", 0.35, 0.0, 3.0) else "stop_or_prepare_blockers_before_push",
        "reasons": sorted(set(reasons)),
        "policy": "Feasibility is diagnostic/priority-setting only; target LUFS never overrides TP/LRA/crest/export locks.",
    }


def _v62_8_classify_mastering_blockers(before: dict[str, Any] | None, decision: dict[str, Any] | None, mode: str) -> dict[str, Any]:
    a = before if isinstance(before, dict) else {}
    tp = _v62_8_metric(a, "approx_true_peak_dbfs", "true_peak_dbtp", default=None)
    crest = _v62_8_metric(a, "crest_factor_db", default=None)
    lra = _v62_8_metric(a, "lra_lu", "loudness_range_lu", default=None)
    side_low = _v61_ms_value(a, "side_ratio_lowband", 0.0) if isinstance(a, dict) else 0.0
    low_mid_ratio = _v62_8_safe_push_feasibility(a, a, decision, mode).get("low_mid_ratio_proxy", 0.0)
    axes: list[str] = []
    classes: list[str] = []
    if tp is not None and float(tp) > float(FINAL_TRUE_PEAK_CEILING_DB) - 0.35:
        axes.append("true_peak_pressure")
        classes.append("sustained_or_near_ceiling_tp_pressure")
    if float(low_mid_ratio or 0.0) >= _env_float("BUSY_V62_8_LOWMID_RATIO_TRIGGER", 0.42, 0.1, 0.9):
        axes.append("low_mid_bottleneck")
        classes.append("low_mid_buildup_120_500")
    if side_low > _env_float("BUSY_V62_8_SIDE_LOW_TRIGGER", 0.18, 0.02, 0.8):
        axes.append("lf_side_or_low_end_headroom_cost")
        classes.append("side_low_headroom_pressure")
    if lra is not None and float(lra) < _env_float("BUSY_V62_8_LRA_NEAR_FLAT_LU", 3.5, 1.0, 8.0):
        axes.append("lra_near_flat_floor")
        classes.append("low_lra_stop_or_eq_focused_finish")
    if crest is not None and float(crest) < _env_float("BUSY_V62_8_CREST_WARNING_DB", 8.0, 4.0, 14.0):
        axes.append("crest_transient_loss")
        classes.append("crest_push_budget_limited")
    return {
        "schema_version": "busy_v62_8_true_peak_pressure_classifier_v8_5_3_62_8",
        "active_axes": sorted(set(axes)),
        "pressure_classes": sorted(set(classes)),
        "metrics": {
            "true_peak_dbtp": round(float(tp), 3) if tp is not None else None,
            "crest_factor_db": round(float(crest), 3) if crest is not None else None,
            "lra_lu": round(float(lra), 3) if lra is not None else None,
            "low_mid_ratio_proxy": round(float(low_mid_ratio or 0.0), 4),
            "side_low_ratio": round(float(side_low), 4),
        },
    }


def _v62_8_append_move(plan: dict[str, Any], key: str, move: dict[str, Any], cap_db: float) -> None:
    try:
        gain = float(move.get("gain_db", 0.0) or 0.0)
        cap = abs(float(cap_db))
        move = copy.deepcopy(move)
        move["gain_db"] = round(float(max(-cap, min(cap, gain))), 3)
        plan.setdefault(key, [])
        if isinstance(plan.get(key), list):
            plan[key].append(move)
    except Exception:
        pass


def _attach_v62_8_mastering_blocker_preparation(before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attach conservative, blocker-specific mastering prep before v61/v60.

    v62.8 does not chase loudness.  It classifies why loudness is not pushable,
    adds small corrective/dynamic-first preparation moves, and exposes a refined
    feasibility report for later guards and telemetry.
    """
    d = copy.deepcopy(decision if isinstance(decision, dict) else {})
    enabled = _env_bool("BUSY_V62_8_MASTERING_BLOCKER_PREP", True)
    feasibility = _v62_8_safe_push_feasibility(before, before, d, mode)
    classifier = _v62_8_classify_mastering_blockers(before, d, mode)
    axes = set(classifier.get("active_axes") or [])
    report: dict[str, Any] = {
        "schema_version": "busy_v62_8_mastering_blocker_preparation_v8_5_3_62_8",
        "enabled": bool(enabled),
        "active": False,
        "classifier": classifier,
        "commercial_loudness_feasibility": feasibility,
        "actions": [],
        "micro_ab_candidate_families": [],
        "policy": "Prepare true-peak/low-mid/LF/crest/LRA blockers before loudness push; corrective and dynamic-first only; no target-LUFS command.",
    }
    if not enabled:
        report["reason"] = "disabled_by_env"
        d["v62_8_mastering_blocker_preparation"] = report
        return d, report
    spectral = d.get("v58_spectral_conditioning_plan") if isinstance(d.get("v58_spectral_conditioning_plan"), dict) else {}
    spectral = copy.deepcopy(spectral) if spectral else {"schema_version": "busy_v58_spectral_conditioning_v8_5_3_58_1", "enabled": True, "active": False, "render_affecting": False, "static_moves": [], "dynamic_moves": [], "reasons": []}
    lf = d.get("v59_lf_crest_source_conditioning_plan") if isinstance(d.get("v59_lf_crest_source_conditioning_plan"), dict) else {}
    lf = copy.deepcopy(lf) if lf else {"schema_version": "busy_v59_lf_crest_source_conditioning_v8_5_3_59_2", "enabled": True, "active": False, "render_affecting": False, "static_moves": [], "dynamic_moves": [], "time_domain_actions": [], "stereo_side_actions": [], "reasons": []}
    for _p in (spectral, lf):
        for k in ("static_moves", "dynamic_moves", "time_domain_actions", "stereo_side_actions", "reasons", "actions"):
            _p.setdefault(k, [])
    strength = _env_float("BUSY_V62_8_BLOCKER_PREP_STRENGTH", 1.0, 0.2, 2.0)
    if "low_mid_bottleneck" in axes:
        fam = "low_mid_decongestion"
        report["micro_ab_candidate_families"].append(fam)
        _v62_8_append_move(spectral, "dynamic_moves", {"type": "bell", "freq": 250, "gain_db": -0.55 * strength, "q": 0.85, "target": "mid", "source": "v62_8_low_mid_bottleneck_dynamic_decongestion"}, cap_db=1.35)
        _v62_8_append_move(spectral, "dynamic_moves", {"type": "bell", "freq": 410, "gain_db": -0.38 * strength, "q": 0.95, "target": "stereo", "source": "v62_8_low_mid_masking_release"}, cap_db=0.95)
        report["actions"].append({"family": fam, "axis": "low_mid_bottleneck", "action": "dynamic_eq_150_500_decongestion"})
    if "lf_side_or_low_end_headroom_cost" in axes or "true_peak_pressure" in axes:
        fam = "lf_envelope_true_peak_relief"
        report["micro_ab_candidate_families"].append(fam)
        _v62_8_append_move(lf, "dynamic_moves", {"type": "bell", "freq": 58, "gain_db": -0.34 * strength, "q": 0.78, "target": "stereo", "source": "v62_8_lf_envelope_limiter_friendly_conditioning"}, cap_db=0.85)
        _v62_8_append_move(lf, "dynamic_moves", {"type": "bell", "freq": 96, "gain_db": -0.28 * strength, "q": 0.86, "target": "mid", "source": "v62_8_bass_body_headroom_smoothing"}, cap_db=0.70)
        _v62_8_append_move(lf, "dynamic_moves", {"type": "bell", "freq": 118, "gain_db": -0.24 * strength, "q": 0.85, "target": "side", "source": "v62_8_side_low_mono_anchor_proxy"}, cap_db=0.60)
        report["actions"].append({"family": fam, "axis": "true_peak_pressure_or_lf_cost", "action": "lf_envelope_cleanup_side_low_anchor"})
    if "crest_transient_loss" in axes:
        fam = "crest_preserving_push_prep"
        report["micro_ab_candidate_families"].append(fam)
        # Do not boost.  Reduce broad dynamics pressure and allow v61/v60 to preserve transient room.
        v59_2 = lf.get("v59_2_program_dependent_dynamics_stereo_side") if isinstance(lf.get("v59_2_program_dependent_dynamics_stereo_side"), dict) else {}
        controls = v59_2.get("program_dynamics_controls") if isinstance(v59_2.get("program_dynamics_controls"), dict) else {}
        controls = copy.deepcopy(controls) if controls else {}
        controls["multiband_amount_scale"] = min(float(controls.get("multiband_amount_scale", 1.0) or 1.0), round(0.90 / max(0.7, strength), 4))
        controls["dynamic_eq_intensity_scale"] = min(float(controls.get("dynamic_eq_intensity_scale", 1.0) or 1.0), 0.94)
        controls["upward_density_scale"] = min(float(controls.get("upward_density_scale", 1.0) or 1.0), 0.88)
        controls["v62_8_crest_preserving_density_guard"] = True
        v59_2["program_dynamics_controls"] = controls
        v59_2["active"] = True
        v59_2["render_affecting"] = True
        lf["v59_2_program_dependent_dynamics_stereo_side"] = v59_2
        report["actions"].append({"family": fam, "axis": "crest_transient_loss", "action": "reduce_broad_density_before_push"})
    if "lra_near_flat_floor" in axes:
        fam = "lra_safe_finish_routing"
        report["micro_ab_candidate_families"].append(fam)
        report["actions"].append({"family": fam, "axis": "lra_near_flat_floor", "action": "eq_focused_finish_stop_and_accept_if_needed"})
    active = bool(report["actions"])
    for _p in (spectral, lf):
        if active:
            _p["enabled"] = True
            _p["active"] = True
            _p["render_affecting"] = True
            _p.setdefault("reasons", []).append("v62_8_mastering_blocker_preparation")
        _p["dynamic_move_count"] = len(_p.get("dynamic_moves") or [])
        _p["static_move_count"] = len(_p.get("static_moves") or [])
        if _p is lf:
            _p["time_domain_action_count"] = len(_p.get("time_domain_actions") or [])
            _p["stereo_side_action_count"] = len(_p.get("stereo_side_actions") or [])
    report["active"] = active
    report["action_count"] = len(report.get("actions") or [])
    report["micro_ab_candidate_families"] = sorted(set(report.get("micro_ab_candidate_families") or []))
    d["v58_spectral_conditioning_plan"] = spectral
    d["v59_lf_crest_source_conditioning_plan"] = lf
    d["v62_8_mastering_blocker_preparation"] = report
    flags = list(d.get("processing_flags") or []) if isinstance(d.get("processing_flags"), list) else []
    if "v62_8_mastering_blocker_preparation" not in flags:
        flags.append("v62_8_mastering_blocker_preparation")
    d["processing_flags"] = flags
    return d, report

def _v61_render_measured_feedback_optimizer(
    staged: np.ndarray,
    sr: int,
    decision: dict[str, Any],
    prepare_report: dict[str, Any],
    mode: str,
    character_scale: float,
    first_audio: np.ndarray,
    first_analysis: dict[str, Any],
    first_proc: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], list[str], dict[str, Any]]:
    """Measured feedback controller for v58/v59/v59.2 source conditioning.

    Unlike the legacy parallel candidate picker, this loop starts from Pass 0,
    measures actual blockers, applies a small blocker-specific decision overlay,
    rerenders from the same staged source, and accepts only measured improvement.
    """
    enabled = _env_bool("BUSY_V61_RENDER_MEASURED_FEEDBACK_OPTIMIZER", True)
    before = prepare_report.get("input_analysis", {}) if isinstance(prepare_report, dict) else {}
    if not enabled:
        reasons = _safety_risk(before, first_analysis, decision)
        return first_audio, first_analysis, first_proc, reasons, {"enabled": False, "active": False, "reason": "disabled_by_env"}
    try:
        max_passes = int(_env_float("BUSY_V61_FEEDBACK_MAX_PASSES", 6, 1, 6))
    except Exception:
        max_passes = 6
    requested_max_passes = int(max_passes)
    v631621_runtime_cap: dict[str, Any] = {"active": False, "requested_max_passes": int(requested_max_passes), "effective_max_passes": int(max_passes)}
    try:
        duration_sec = float(len(staged)) / float(sr or 1)
        v631621_runtime_cap["duration_sec"] = round(float(duration_sec), 3)
        if _env_bool("BUSY_V631621_V61_RUNTIME_CAP", True):
            if duration_sec >= _env_float("BUSY_V631621_V61_CAP_LONG_TRACK_SEC", 90.0, 10.0, 900.0):
                cap = int(_env_float("BUSY_V631621_V61_LONG_TRACK_MAX_PASSES", 2, 1, 6))
            elif duration_sec >= _env_float("BUSY_V631621_V61_CAP_MEDIUM_TRACK_SEC", 30.0, 5.0, 240.0):
                cap = int(_env_float("BUSY_V631621_V61_MEDIUM_TRACK_MAX_PASSES", 3, 1, 6))
            else:
                cap = max_passes
            if cap < max_passes:
                max_passes = max(1, int(cap))
                v631621_runtime_cap.update({
                    "active": True,
                    "effective_max_passes": int(max_passes),
                    "reason": "full_track_feedback_uses_bounded_pass_count_until_stress_window_proxy_is_available",
                })
    except Exception as exc:
        v631621_runtime_cap.update({"active": False, "reason": "runtime_cap_exception", "error": str(exc)[:180]})
    epsilon = _env_float("BUSY_V61_FEEDBACK_EPSILON", 0.05, 0.0, 4.0)
    history: list[dict[str, Any]] = []
    base_blockers = _v61_blockers(before, first_analysis, decision, mode)
    base_score, base_components, base_warnings = _v61_score_pass(before, first_analysis, decision, mode, applied_action_count=0)
    base_hard = _v61_hard_reject(before, first_analysis, base_warnings)
    # v62.9.6.3: when upstream v57/v62.8 already proved that safe loudness
    # push budget is zero and the baseline is blocked by hard true-peak/crest
    # pressure, repeated v61 trust-region attempts only waste runtime and are
    # predictably rejected.  Exit as an explicit diagnostic skip instead of
    # rolling through four hard rejects.
    try:
        prep = (decision or {}).get('v62_8_mastering_blocker_preparation') if isinstance(decision, dict) else {}
        feas = prep.get('commercial_loudness_feasibility') if isinstance(prep, dict) else {}
        safe_budget = float(feas.get('safe_push_budget_db') if isinstance(feas, dict) and feas.get('safe_push_budget_db') is not None else 999.0)
        feas_decision = str(feas.get('decision') or '') if isinstance(feas, dict) else ''
        blocker_axes = {str(b.get('axis') or '') for b in base_blockers if isinstance(b, dict)}
        hard_set = {str(x) for x in (base_hard or [])}
        skip_enabled = _env_bool('BUSY_V62963_V61_SKIP_ZERO_SAFE_PUSH_HARD_BLOCKERS', False)
        if skip_enabled and safe_budget <= 0.001 and ('true_peak_pressure' in blocker_axes or 'hard_true_peak_pressure' in hard_set) and ('crest_transient_loss' in blocker_axes or 'lra_near_flat_floor' in blocker_axes or 'stop_or_prepare' in feas_decision):
            history = [{
                'pass_index': 0,
                'status': 'BASELINE',
                'score': base_score,
                'score_components': base_components,
                'blockers_detected': base_blockers,
                'warnings': base_warnings,
                'hard_reject_reasons': base_hard,
                'metrics': _analysis_summary_for_loudness(first_analysis),
                'applied_update': {'actions': [], 'loudness_offset_db': 0.0, 'mb_amount_scale': 1.0, 'character_scale_multiplier': 1.0},
            }, {
                'pass_index': 1,
                'status': 'SKIPPED',
                'reason': 'v62963_safe_push_zero_hard_blocker_skip',
                'safe_push_budget_db': round(float(safe_budget), 3),
                'blocker_axes': sorted(blocker_axes),
                'hard_reject_reasons': sorted(hard_set),
            }]
            reasons = sorted(set(base_warnings + [b.get('axis') for b in base_blockers if isinstance(b, dict) and b.get('axis')]))
            return first_audio, first_analysis, first_proc, reasons, {
                'schema_version': 'busy_v61_render_measured_feedback_optimizer_v8_5_3_62_9_6_6_2',
                'enabled': True,
                'active': False,
                'attempted': False,
                'reason': 'v62963_safe_push_zero_hard_blocker_skip',
                'selected_pass': 0,
                'selected_score': round(float(base_score), 4),
                'base_score': round(float(base_score), 4),
                'score_delta': 0.0,
                'accepted_pass_count': 0,
                'rollback_count': 0,
                'max_passes': int(max_passes),
                'requested_max_passes': int(requested_max_passes),
                'v631621_runtime_cap': v631621_runtime_cap,
                'epsilon': round(float(epsilon), 4),
                'stop_reason': 'v62963_safe_push_zero_hard_blocker_skip',
                'selected_blockers_remaining': base_blockers,
                'safe_push_budget_db': round(float(safe_budget), 3),
                'passes': history,
                'policy': 'v62.9.6.3 skips v61 feedback attempts when v57/v62.8 safe-push budget is zero and hard TP/crest/LRA blockers would force rejection; no DSP behavior changes beyond avoiding rejected rerenders.',
            }
    except Exception:
        pass
    best_audio = first_audio
    best_analysis = first_analysis
    best_proc = first_proc
    best_decision = copy.deepcopy(decision if isinstance(decision, dict) else {})
    current_decision = copy.deepcopy(best_decision)
    best_score = base_score
    best_pass = 0
    best_warnings = sorted(set(base_warnings))
    history.append({
        "pass_index": 0,
        "status": "BASELINE",
        "score": base_score,
        "score_components": base_components,
        "blockers_detected": base_blockers,
        "warnings": base_warnings,
        "hard_reject_reasons": base_hard,
        "metrics": _analysis_summary_for_loudness(first_analysis),
        "applied_update": {"actions": [], "loudness_offset_db": 0.0, "mb_amount_scale": 1.0, "character_scale_multiplier": 1.0},
    })
    if not base_blockers:
        return best_audio, best_analysis, best_proc, best_warnings, {
            "schema_version": "busy_v61_render_measured_feedback_optimizer_v8_5_3_61_2",
            "enabled": True,
            "active": False,
            "reason": "no_measured_blockers",
            "selected_pass": 0,
            "selected_score": best_score,
            "passes": history,
            "policy": "No feedback action without measured blockers.",
        }

    trust_scale = 1.0
    current_blockers = base_blockers
    rollback_count = 0
    accepted_count = 0
    for pass_idx in range(1, max_passes + 1):
        if not current_blockers:
            break
        try:
            # v61.2 hotfix: build the next attempt from the latest accepted
            # decision state, not from the original base decision.  Without this,
            # successful pass-N source-conditioning changes were not carried into
            # pass N+1, so the loop behaved like a shallow candidate selector rather
            # than a measured convergence controller.
            d2, update = _v61_feedback_decision(current_decision, current_blockers, pass_idx, trust_scale)
            if not update.get("actions"):
                history.append({"pass_index": pass_idx, "status": "STOP", "reason": "no_actionable_blockers", "blockers_detected": current_blockers})
                break
            char_mul = float(update.get("character_scale_multiplier", 1.0) or 1.0)
            loud_off = float(update.get("loudness_offset_db", 0.0) or 0.0)
            mb_scale = float(update.get("mb_amount_scale", 1.0) or 1.0)
            cand_raw, proc = _render_once(staged, sr, d2, mode, character_scale=float(character_scale) * char_mul, loudness_offset_db=loud_off, mb_amount_scale=mb_scale)
            cand_pre_safety_analysis = analyze_audio_fast_qc(cand_raw, sr, true_peak_oversample=_qc_oversample_factor())
            cand, cand_analysis, cand_post_safety_qc = _v63162_post_safety_limit_candidate(
                cand_raw,
                sr,
                limiter_cfg=(d2.get("limiter", {}) if isinstance(d2, dict) and isinstance(d2.get("limiter"), dict) else {}),
                label=f"v61_pass_{pass_idx}",
            )
            cand_blockers = _v61_blockers(before, cand_analysis, d2, mode)
            cand_score, cand_components, cand_warnings = _v61_score_pass(before, cand_analysis, d2, mode, applied_action_count=len(update.get("actions") or []))
            hard_reject = _v61_hard_reject(before, cand_analysis, cand_warnings)
            improved = bool((not hard_reject) and cand_score > best_score + epsilon)
            status = "ACCEPTED" if improved else ("REJECTED_HARD" if hard_reject else "REJECTED_SOFT")
            pass_reasons = _v61_feedback_status_reasons(status, cand_score, best_score, epsilon, hard_reject, cand_blockers, cand_components, cand_warnings, update)
            row = {
                "pass_index": pass_idx,
                "status": status,
                "score": cand_score,
                "score_delta_vs_best": round(float(cand_score - best_score), 4),
                "score_components": cand_components,
                "blockers_detected": cand_blockers,
                "warnings": cand_warnings,
                "hard_reject_reasons": hard_reject,
                "reject_reasons": pass_reasons if status != "ACCEPTED" else [],
                "decision_reasons": pass_reasons,
                "applied_update": update,
                "metrics": _analysis_summary_for_loudness(cand_analysis),
                "pre_safety_metrics": _analysis_summary_for_loudness(cand_pre_safety_analysis),
                "v63162_post_safety_qc": cand_post_safety_qc,
                "rollback_source": best_pass if not improved else None,
            }
            history.append(row)
            _debug("v61_feedback_pass", pass_index=pass_idx, status=status, score=cand_score, best_score=best_score, blockers=[b.get("axis") for b in cand_blockers], actions=[a.get("axis") for a in (update.get("actions") or [])])
            if improved:
                best_audio = cand
                best_analysis = cand_analysis
                best_proc = proc
                best_score = cand_score
                best_pass = pass_idx
                best_warnings = sorted(set(cand_warnings))
                best_decision = copy.deepcopy(d2)
                current_decision = copy.deepcopy(best_decision)
                current_blockers = cand_blockers
                accepted_count += 1
                trust_scale = max(0.30, trust_scale * 0.90)
                if abs(float(row.get("score_delta_vs_best", 0.0) or 0.0)) < epsilon:
                    break
            else:
                rollback_count += 1
                trust_scale *= 0.50
                if trust_scale < _env_float("BUSY_V61_MIN_TRUST_SCALE", 0.08, 0.03, 0.8):
                    history.append({"pass_index": pass_idx, "status": "STOP", "reason": "trust_region_exhausted", "trust_scale": round(float(trust_scale), 4)})
                    break
                # Retry from the same best accepted baseline with smaller changes;
                # do not stack rejected parameter changes.
                current_decision = copy.deepcopy(best_decision)
                current_blockers = current_blockers[:]
                del cand
            gc.collect()
        except Exception as exc:
            history.append({"pass_index": pass_idx, "status": "EXCEPTION", "error_type": type(exc).__name__, "error": str(exc)[:260], "rollback_source": best_pass})
            rollback_count += 1
            trust_scale *= 0.50
            if trust_scale < 0.20:
                break
    selected_blockers = _v61_blockers(before, best_analysis, best_decision, mode)
    stop_reason = "max_passes_or_no_better_candidate"
    try:
        if not current_blockers:
            stop_reason = "no_remaining_actionable_blockers"
        elif trust_scale < _env_float("BUSY_V61_MIN_TRUST_SCALE", 0.08, 0.03, 0.8):
            stop_reason = "trust_region_exhausted"
        elif accepted_count == 0 and rollback_count > 0:
            stop_reason = "all_feedback_attempts_rejected"
        elif accepted_count > 0:
            stop_reason = "best_safe_feedback_state_selected"
    except Exception:
        pass
    report = {
        "schema_version": "busy_v61_render_measured_feedback_optimizer_v8_5_3_63_1_6_2_post_safety_qc",
        "enabled": True,
        "active": bool(accepted_count > 0),
        "attempted": bool(len(history) > 1),
        "selected_pass": int(best_pass),
        "selected_score": round(float(best_score), 4),
        "base_score": round(float(base_score), 4),
        "score_delta": round(float(best_score - base_score), 4),
        "accepted_pass_count": int(accepted_count),
        "rollback_count": int(rollback_count),
        "max_passes": int(max_passes),
        "requested_max_passes": int(requested_max_passes),
        "v631621_runtime_cap": v631621_runtime_cap,
        "epsilon": round(float(epsilon), 4),
        "min_trust_scale": round(float(_env_float("BUSY_V61_MIN_TRUST_SCALE", 0.08, 0.03, 0.8)), 4),
        "stop_reason": stop_reason,
        "selected_blockers_remaining": selected_blockers,
        "passes": history,
        "accepted_state_carry_forward": True,
        "calibration_basis": "v61.2 defaults follow report examples where available: max_passes=6, epsilon=0.05, LRA hard floor=2.5 LU with source-already-flat exception; other weights remain conservative engineering defaults.",
        "policy": "Measured blockers update v58/v59/v59.2 source-conditioning overlays only within trust-region limits; candidates are scored after final safety limiting/TP attenuation so pre-safety TP spikes do not hard-reject otherwise valid polish; accepted pass states are carried forward; rejected changes are rolled back; no-boost, dynamic-first, rollback-on-damage, target LUFS never overrides damage gates.",
    }
    return best_audio, best_analysis, best_proc, best_warnings, report


def _prepare_optimizer_candidate(y: np.ndarray, sr: int, ceiling_db: float, name: str, decision: dict[str, Any]) -> np.ndarray:
    """Create one lightweight candidate from an already limited master."""
    out = y
    if name == "punch_restore":
        out = transient_micro_expander(out, sr, amount=0.035)
        out = lookahead_limiter(out, sr, ceiling_db=ceiling_db, lookahead_ms=0.8, release_ms=130)
    elif name == "codec_smooth":
        moves = [
            {"type": "high_shelf", "freq": 9800, "gain_db": -0.22, "q": 0.70},
            {"type": "bell", "freq": 4200, "gain_db": -0.10, "q": 1.10},
        ]
        out = apply_eq_moves(out, sr, moves, scale=1.0)
        out = lookahead_limiter(out, sr, ceiling_db=ceiling_db, lookahead_ms=1.0, release_ms=145)
    elif name == "micro_loud":
        out = apply_gain_db(out, 0.22)
        out = lookahead_limiter(out, sr, ceiling_db=ceiling_db, lookahead_ms=1.0, release_ms=150)
    elif name == "low_stability":
        out = ms_low_mono(out, sr, cutoff_hz=125, amount=0.22)
        out = lookahead_limiter(out, sr, ceiling_db=ceiling_db, lookahead_ms=1.0, release_ms=150)
    out, _, _ = _attenuate_to_true_peak_ceiling(out, ceiling_db, sr=sr, oversample=_qc_oversample_factor())
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _multi_pass_auto_optimizer(final: np.ndarray, sr: int, before: dict[str, Any], decision: dict[str, Any], mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Try a few safe finishing candidates and keep the best.

    Candidates are rendered one at a time, fast-QC'd, and discarded unless selected.
    """
    if not _env_bool("BUSY_MULTI_PASS_OPTIMIZER", True):
        return final, {"active": False, "reason": "disabled"}
    max_candidates = int(_env_float("BUSY_OPTIMIZER_MAX_CANDIDATES", 4, 1, 4))
    allowed_names = ["base", "punch_restore", "codec_smooth", "micro_loud", "low_stability"][: max_candidates + 1]
    reports: list[dict[str, Any]] = []

    base_audio = final.astype(np.float32, copy=False)
    base_analysis = analyze_audio_fast_qc(base_audio, sr, true_peak_oversample=_qc_oversample_factor())
    base_score, base_warnings = _master_score_for_optimizer(before, base_analysis, decision, mode)
    best_audio = base_audio
    best_name = "base"
    best_score = base_score
    best_analysis = base_analysis
    reports.append({
        "name": "base",
        "score": base_score,
        "warnings": base_warnings,
        "lufs": base_analysis.get("integrated_lufs"),
        "true_peak": base_analysis.get("approx_true_peak_dbfs"),
        "crest": base_analysis.get("crest_factor_db"),
    })
    _debug("optimizer_candidate", name="base", score=base_score, warnings=base_warnings, lufs=base_analysis.get("integrated_lufs"), crest=base_analysis.get("crest_factor_db"))

    ms = base_analysis.get("ms", {}) if isinstance(base_analysis, dict) else {}
    q = base_analysis.get("quality_indices", {}) if isinstance(base_analysis, dict) else {}
    low_side = float(ms.get("side_ratio_lowband", 0) or 0)
    high_side = float(ms.get("side_ratio_8k_14k", 0) or 0)
    crest = float(base_analysis.get("crest_factor_db", 0) or 0)
    lufs = float(base_analysis.get("integrated_lufs", -99) or -99)
    air = float(q.get("air_index_db", 0) or 0)
    try:
        _opt_profile = _commercial_loudness_finish_profile(decision or {}, mode)
    except Exception:
        _opt_profile = {}
    opt_target_lufs = _v60_float_metric(_opt_profile, "target_lufs", default=None) if isinstance(_opt_profile, dict) else None
    opt_floor_lufs = _v60_float_metric(_opt_profile, "commercial_floor_lufs", default=None) if isinstance(_opt_profile, dict) else None
    opt_overshoot_tol = _env_float("BUSY_OPTIMIZER_TARGET_OVERSHOOT_TOLERANCE_DB", 0.45, 0.0, 2.0)
    opt_max_lufs_delta = _env_float("BUSY_OPTIMIZER_MAX_LUFS_DELTA_DB", 0.75, 0.05, 3.0)

    ordered: list[str] = []
    if crest < 9.0 or "transient_loss_delta" in base_warnings:
        ordered.append("punch_restore")
    if high_side > 0.55 or air > 0.0 or "side_air_delta_excess" in base_warnings:
        ordered.append("codec_smooth")
    if lufs < -10.2 and not base_warnings:
        ordered.append("micro_loud")
    if low_side > 0.10:
        ordered.append("low_stability")
    for fallback in ["punch_restore", "codec_smooth", "micro_loud", "low_stability"]:
        if fallback not in ordered:
            ordered.append(fallback)
    ordered = [x for x in ordered if x in allowed_names and x != "base"][:max_candidates]

    for name in ordered:
        try:
            _debug("optimizer_start_candidate", name=name)
            cand = _prepare_optimizer_candidate(base_audio, sr, FINAL_TRUE_PEAK_CEILING_DB, name, decision)
            cand_analysis = analyze_audio_fast_qc(cand, sr, true_peak_oversample=_qc_oversample_factor())
            score, warnings = _master_score_for_optimizer(before, cand_analysis, decision, mode)
            cand_lufs_val = _v60_float_metric(cand_analysis, "integrated_lufs", default=None)
            reject_reasons: list[str] = []
            if cand_lufs_val is not None and opt_target_lufs is not None and float(cand_lufs_val) > float(opt_target_lufs) + float(opt_overshoot_tol):
                reject_reasons.append("optimizer_candidate_overshoots_target_lufs")
            elif cand_lufs_val is not None and opt_floor_lufs is not None and float(cand_lufs_val) > float(opt_floor_lufs) + _env_float("BUSY_OPTIMIZER_FLOOR_OVERSHOOT_TOLERANCE_DB", 1.25, 0.2, 3.0):
                reject_reasons.append("optimizer_candidate_overshoots_commercial_floor_margin")
            if cand_lufs_val is not None and float(cand_lufs_val) > float(lufs) + float(opt_max_lufs_delta):
                reject_reasons.append("optimizer_candidate_lufs_jump_exceeds_gate")
            rep = {
                "name": name,
                "score": score,
                "warnings": warnings,
                "lufs": cand_analysis.get("integrated_lufs"),
                "true_peak": cand_analysis.get("approx_true_peak_dbfs"),
                "crest": cand_analysis.get("crest_factor_db"),
                "rejected_by_optimizer_loudness_gate": bool(reject_reasons),
                "optimizer_loudness_reject_reasons": reject_reasons,
            }
            reports.append(rep)
            _debug("optimizer_done_candidate", **rep)
            if reject_reasons:
                del cand
            elif score > best_score + 0.75:
                best_audio = cand
                best_name = name
                best_score = score
                best_analysis = cand_analysis
            else:
                del cand
        except Exception as exc:
            reports.append({"name": name, "error": str(exc)})
            _debug("optimizer_candidate_failed", name=name, error=str(exc))

    return best_audio, {
        "active": True,
        "selected": best_name,
        "selected_score": best_score,
        "base_score": base_score,
        "candidates": reports,
        "optimizer_loudness_gate": {
            "target_lufs": round(float(opt_target_lufs), 3) if opt_target_lufs is not None else None,
            "commercial_floor_lufs": round(float(opt_floor_lufs), 3) if opt_floor_lufs is not None else None,
            "target_overshoot_tolerance_db": round(float(opt_overshoot_tol), 3),
            "max_lufs_delta_db": round(float(opt_max_lufs_delta), 3),
            "policy": "post-limiter optimizer candidates may improve tone/crest/codec safety but may not secretly become a loudness push",
        },
        "selected_fast_qc": {
            "lufs": best_analysis.get("integrated_lufs"),
            "true_peak": best_analysis.get("approx_true_peak_dbfs"),
            "crest": best_analysis.get("crest_factor_db"),
            "correlation": best_analysis.get("correlation"),
        },
    }




def _commercial_family_text(decision: dict[str, Any]) -> str:
    """Collect public genre/family/profile hints for commercial-loudness policy."""
    if not isinstance(decision, dict):
        return ""
    parts: list[str] = []
    parts.append(_profile_text(decision))
    for key in ("primary_profile", "selected_family", "selected_profile", "detected_style", "mastering_intent"):
        parts.append(str(decision.get(key, "") or ""))
    gc = decision.get("genre_chain", {}) if isinstance(decision.get("genre_chain", {}), dict) else {}
    for key in ("dominant_family", "selected_family", "selected_profile"):
        parts.append(str(gc.get(key, "") or ""))
    for item in (decision.get("profile_blend") or []):
        if isinstance(item, dict):
            parts.append(str(item.get("profile", "") or ""))
            parts.append(str(item.get("role", "") or ""))
    return " ".join(parts).lower().replace("-", "_")



def _lufs_range_to_pair(value: Any) -> tuple[float, float] | None:
    """Return a LUFS range as (quiet_edge, loud_edge).

    Private reference DB assets have used both list forms such as
    [-8.5, -6.5] and dict forms such as {"min": -8.5, "max": -6.5}.
    For negative LUFS values, the smaller number is the quieter edge and the
    larger number is the hotter/louder edge.
    """
    vals: list[float] = []
    if isinstance(value, dict):
        keys = ["min", "max", "low", "high", "lower", "upper", "quiet", "hot"]
        for k in keys:
            if k in value:
                try:
                    vals.append(float(value.get(k)))
                except Exception:
                    pass
        # Some private DB variants store nested ranges or arrays under a value key.
        for k in ["range", "integrated_lufs_range", "lufs_range"]:
            if k in value and len(vals) < 2:
                nested = _lufs_range_to_pair(value.get(k))
                if nested is not None:
                    return nested
    elif isinstance(value, (list, tuple)):
        for x in value[:2]:
            try:
                vals.append(float(x))
            except Exception:
                pass
    if len(vals) < 2:
        return None
    a, b = vals[0], vals[1]
    if not (-30.0 <= a <= -3.0 and -30.0 <= b <= -3.0):
        return None
    quiet, loud = (a, b) if a <= b else (b, a)
    if loud - quiet < 0.05:
        return None
    return float(quiet), float(loud)


def _commercial_loudness_range_sources(decision: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect loudness target ranges already loaded from the private DB overlay.

    The worker downloads the private reference bundle from Supabase, then
    reference_v7 writes the chosen mode_profile into decision.reference_v7.  The
    commercial finish pass should use that DB-authored range instead of a hard
    coded family table.  decision.targets is kept as a fallback because it may be
    safety-adjusted after profile selection.
    """
    sources: list[dict[str, Any]] = []
    detailed = decision.get("detailed_genre_decision", {}) if isinstance(decision.get("detailed_genre_decision", {}), dict) else {}
    owner = detailed.get("loudness_owner", {}) if isinstance(detailed.get("loudness_owner", {}), dict) else {}
    pair = _lufs_range_to_pair(owner.get("resolved_lufs_range"))
    if pair is not None and owner.get("profile"):
        sources.append({
            "source": "detailed_genre_decision.loudness_owner.resolved_lufs_range",
            "range": [round(pair[0], 3), round(pair[1], 3)],
            "selected_profile": owner.get("profile"),
            "family": owner.get("family"),
            "role": owner.get("role"),
            "confidence": owner.get("confidence"),
            "safer_floor_lufs": owner.get("safer_floor_lufs"),
            "qc_gate_required": True,
            "is_automatic_target": False,
            "priority": 12,
        })
    rv7 = decision.get("reference_v7", {}) if isinstance(decision.get("reference_v7", {}), dict) else {}
    mode_profile = rv7.get("mode_profile", {}) if isinstance(rv7.get("mode_profile", {}), dict) else {}
    for key in ["integrated_lufs_range", "commercial_lufs_range", "mastering_lufs_range"]:
        pair = _lufs_range_to_pair(mode_profile.get(key))
        if pair is not None:
            sources.append({
                "source": f"reference_v7.mode_profile.{key}",
                "range": [round(pair[0], 3), round(pair[1], 3)],
                "selected_profile": rv7.get("selected_profile"),
                "family": rv7.get("family"),
                "delivery_mode": rv7.get("delivery_mode"),
                "priority": 10,
            })
            break
    # If the reference overlay was not available, fall back to the current DSP
    # targets rather than an internal genre-name table.
    targets = decision.get("targets", {}) if isinstance(decision.get("targets", {}), dict) else {}
    pair = _lufs_range_to_pair(targets.get("integrated_lufs_range"))
    if pair is not None:
        sources.append({
            "source": "decision.targets.integrated_lufs_range",
            "range": [round(pair[0], 3), round(pair[1], 3)],
            "target_lufs": targets.get("integrated_lufs_target"),
            "priority": 5,
        })
    return sources



def _bamix_handoff_policy_from_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(decision, dict):
        return {}
    direct = decision.get("busy_auto_mixing_mastering_handoff_policy")
    if isinstance(direct, dict) and direct.get("active"):
        return direct
    ai = decision.get("auto_intensity_policy") if isinstance(decision.get("auto_intensity_policy"), dict) else {}
    hp = ai.get("bamix_premaster_handoff_policy") if isinstance(ai, dict) else None
    if isinstance(hp, dict) and hp.get("active"):
        return hp
    iam = decision.get("input_assist_modes") if isinstance(decision.get("input_assist_modes"), dict) else {}
    bamix = iam.get("busy_auto_mixing") if isinstance(iam.get("busy_auto_mixing"), dict) else {}
    if bamix.get("premaster_used_for_mastering") or bamix.get("mastering_input_source") == "busy_auto_mixing_premaster":
        return {"active": True, "inferred_from_input_assist_modes": True}
    flags = decision.get("processing_flags") if isinstance(decision.get("processing_flags"), list) else []
    if "busy_auto_mixing_premaster_used_for_mastering" in flags:
        return {"active": True, "inferred_from_processing_flags": True}
    return {}


def _v6314_handoff_policy_from_decision(decision: dict[str, Any] | None) -> dict[str, Any]:
    hp = _bamix_handoff_policy_from_decision(decision)
    if not isinstance(hp, dict) or not hp.get("active"):
        return {}
    return hp


def _bamix_v6319_handoff_mode(decision: dict[str, Any] | None) -> str:
    hp = _bamix_handoff_policy_from_decision(decision)
    mode = str(hp.get("handoff_mode") or "").strip().lower()
    if not mode and hp.get("active"):
        profile = str(hp.get("profile") or "").lower()
        if "density" in profile:
            mode = "commercial_density_completion"
        elif "fragile" in profile:
            mode = "fragile_protection"
        elif profile:
            mode = "transparent_polish"
    if not mode and hp.get("active") and _env_bool("BUSY_AUTOMIX_PREMASTER_ACTIVE", False):
        # Runtime env is only a fallback for legacy payloads inside the active
        # BAMix job.  It must not override policy metadata or leak across Cloud
        # Run warm-container jobs.
        mode = str(os.environ.get("BUSY_BAMIX_V6319_HANDOFF_MODE", "") or "").strip().lower()
    if not mode and hp.get("active"):
        mode = "transparent_polish"
    return mode


def _bamix_policy_float(
    policy: dict[str, Any] | None,
    key: str,
    env_name: str,
    default: float,
    lo: float | None = None,
    hi: float | None = None,
) -> float:
    """Resolve BAMix handoff numbers policy-first.

    Older handoff code wrote run-specific values to process env.  Cloud Run can
    reuse warm containers, so env-first lookup can make a later job inherit a
    previous BAMix target/floor/push budget.  The decision/report policy is the
    run authority; env can override only when explicitly requested for manual
    debugging.
    """
    hp = policy if isinstance(policy, dict) else {}
    use_env = _env_bool("BUSY_BAMIX_V6319_ALLOW_ENV_POLICY_OVERRIDE", False)
    raw = os.environ.get(env_name) if use_env and env_name else hp.get(key, default)
    try:
        val = float(raw)
        if not np.isfinite(val):
            val = float(default)
    except Exception:
        val = float(default)
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def _bamix_v6319_density_completion_allowed(decision: dict[str, Any] | None) -> bool:
    return _bamix_v6319_handoff_mode(decision) in {"commercial_density_completion", "guarded_completion"}


def _bamix_v6319_transparent_or_fragile(decision: dict[str, Any] | None) -> bool:
    mode = _bamix_v6319_handoff_mode(decision)
    return mode in {"transparent_polish", "fragile_protection"}


def _analysis_crest_db(analysis: dict[str, Any] | None) -> float | None:
    a = analysis if isinstance(analysis, dict) else {}
    loud = a.get("loudness", {}) if isinstance(a.get("loudness"), dict) else {}
    for key in ("crest_factor_db", "crest_db"):
        v = _airqc_num(a.get(key), None)
        if v is not None:
            return v
        v = _airqc_num(loud.get(key), None)
        if v is not None:
            return v
    return None


def _v6314_crest_loss_db(reference: dict[str, Any] | None, candidate: dict[str, Any] | None) -> float | None:
    ref = _analysis_crest_db(reference)
    cand = _analysis_crest_db(candidate)
    if ref is None or cand is None:
        return None
    return float(ref) - float(cand)


def _v6313_3_lra_warning_semantics(lra_guard: dict[str, Any] | None, adaptive_qc: dict[str, Any] | None, warnings: list[str] | None) -> dict[str, Any]:
    """Report-only semantic split for generic LRA warnings.

    lra_too_flat can mean three different things: the source was already flat,
    the final is below an absolute style floor, or mastering actually reduced
    LRA.  Keep raw warnings, but expose display warnings and categories so debug
    brief/Supabase do not count all cases as the same damage event.
    """
    lg = lra_guard if isinstance(lra_guard, dict) else {}
    aq = adaptive_qc if isinstance(adaptive_qc, dict) else {}
    wr = aq.get("warning_reclassification") if isinstance(aq.get("warning_reclassification"), dict) else {}
    raw_lra = [str(x) for x in (lg.get("raw_warnings_before_adaptive_input_relative_qc") or lg.get("warnings") or [])]
    final_warnings = [str(x) for x in (warnings or [])]
    aq_lra = aq.get("lra") if isinstance(aq.get("lra"), dict) else {}
    classification = str(aq_lra.get("classification") or "unknown")
    relative_damage = [w for w in final_warnings if w in {"mastering_induced_lra_collapse", "lra_loss_relative", "source_flat_plus_additional_lra_loss"}]
    source_inherited = [w for w in final_warnings if w in {"source_lra_already_flat", "lra_flat_but_input_relative_loss_small", "lra_low_input_unavailable"}]
    absolute = [w for w in raw_lra if w in {"lra_too_flat", "lra_too_wide", "lra_reduced_too_much"}]
    display = []
    for w in final_warnings:
        if w == "lra_too_flat" and (relative_damage or source_inherited):
            continue
        display.append(w)
    return {
        "schema_version": "busy_lra_warning_semantics_v8_5_3_63_1_3_3",
        "active": True,
        "classification": classification,
        "raw_lra_guard_warnings": sorted(set(raw_lra)),
        "absolute_threshold_warnings": sorted(set(absolute)),
        "source_inherited_or_monitor_warnings": sorted(set(source_inherited)),
        "input_relative_damage_warnings": sorted(set(relative_damage)),
        "display_warnings": sorted(set(display)),
        "removed_by_adaptive_qc": sorted(set(str(x) for x in (wr.get("removed_warnings", []) or []))) if isinstance(wr, dict) else [],
        "added_by_adaptive_qc": sorted(set(str(x) for x in (wr.get("added_warnings", []) or []))) if isinstance(wr, dict) else [],
        "policy": "Separate absolute LRA floor telemetry from source-inherited flatness and mastering-induced LRA loss; this is report-only.",
    }


def _v6314_repeated_crest_loss_push_cap(report: dict[str, Any] | None, proposed_push: float, decision: dict[str, Any] | None) -> dict[str, Any]:
    hp = _v6314_handoff_policy_from_decision(decision)
    if not hp.get("active") or not _env_bool("BUSY_BAMIX_V6314_REPEAT_CREST_LOSS_CAP", True):
        return {"active": False, "effective_push_db": round(float(proposed_push), 3), "reason": "inactive"}
    attempts = (report or {}).get("render_attempts") if isinstance(report, dict) else []
    attempts = attempts if isinstance(attempts, list) else []
    def _has_crest_loss(rep: dict[str, Any]) -> bool:
        vals: list[str] = []
        for key in ("warnings", "hard_blocks", "unresolved_blocker_set"):
            v = rep.get(key) if isinstance(rep, dict) else []
            if isinstance(v, list):
                vals.extend(str(x) for x in v)
        gate = rep.get("v57_damage_aware_candidate_gate") if isinstance(rep, dict) and isinstance(rep.get("v57_damage_aware_candidate_gate"), dict) else {}
        if isinstance(gate.get("blockers"), list):
            vals.extend(str(x) for x in gate.get("blockers"))
        return any("crest_loss_relative" == x or "crest_loss" in x for x in vals)
    hits = [a for a in attempts if isinstance(a, dict) and _has_crest_loss(a)]
    if not hits:
        return {"active": True, "effective_push_db": round(float(proposed_push), 3), "crest_loss_relative_count": 0, "reason": "no_prior_crest_loss_relative"}
    decay = float(hp.get("crest_loss_repeat_push_decay") or _env_float("BUSY_BAMIX_V6314_REPEAT_CREST_LOSS_PUSH_DECAY", 0.72, 0.35, 0.95))
    factor = max(0.25, decay ** len(hits))
    effective = min(float(proposed_push), max(0.0, float(proposed_push) * factor))
    return {
        "schema_version": "busy_v6314_repeated_crest_loss_push_cap",
        "active": True,
        "crest_loss_relative_count": len(hits),
        "proposed_push_db": round(float(proposed_push), 3),
        "effective_push_db": round(float(effective), 3),
        "decay": round(float(decay), 3),
        "reason": "repeated_crest_loss_relative_shrinks_remaining_final_push_cap",
    }


def _v6314_candidate_dynamic_governance(
    reference: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    floor_lufs: float,
    *,
    candidate_label: str = "candidate",
) -> dict[str, Any]:
    hp = _v6314_handoff_policy_from_decision(decision)
    if not hp.get("active"):
        return {"active": False, "accepted": True, "reason": "not_bamix_handoff"}
    crest_loss = _v6314_crest_loss_db(reference, candidate)
    ref_lufs = _airqc_num((reference or {}).get("integrated_lufs"), None)
    cand_lufs = _airqc_num((candidate or {}).get("integrated_lufs"), None)
    ref_met = bool(ref_lufs is not None and float(ref_lufs) >= float(floor_lufs))
    cand_met = bool(cand_lufs is not None and float(cand_lufs) >= float(floor_lufs))
    max_loss = _airqc_num(hp.get("max_crest_loss_db"), _env_float("BUSY_BAMIX_V6314_PUNCH_MAX_CREST_LOSS_DB", 2.85, 1.0, 6.5))
    handoff_mode = _bamix_v6319_handoff_mode(decision)
    if handoff_mode in {"commercial_density_completion", "guarded_completion"}:
        # v63.1.9: crest budget is a telemetry/soft-governance input in density
        # completion.  The actual veto is still handled by the existing v57/v60/
        # perceptual/export gates; do not keep the quieter standard solely because
        # both versions technically crossed a conservative BAMix crest line.
        max_loss = max(float(max_loss or 0.0), _env_float("BUSY_BAMIX_V6319_DENSITY_SOFT_CREST_LOSS_DB", 4.85 if handoff_mode == "commercial_density_completion" else 3.8, 1.5, 7.0))
    soften_margin = _env_float("BUSY_BAMIX_V6314_CREST_SOFTEN_MARGIN_DB", 0.25, 0.0, 0.75)
    punch = bool(hp.get("punch_preserved") or str(hp.get("selected_mix_recipe") or os.environ.get("BUSY_BAMIX_SELECTED_RECIPE", "")).lower() == "punch_preserved")
    near = bool(crest_loss is not None and max_loss is not None and float(crest_loss) >= max(0.0, float(max_loss) - float(soften_margin)))
    over = bool(crest_loss is not None and max_loss is not None and float(crest_loss) > float(max_loss))
    prefer_cleaner = bool(hp.get("prefer_cleaner_quieter_when_target_met", True)) and handoff_mode not in {"commercial_density_completion", "guarded_completion"}
    reject_cleaner = bool(prefer_cleaner and ref_met and cand_met and near)
    return {
        "schema_version": "busy_v6314_candidate_dynamic_governance",
        "active": True,
        "candidate_label": candidate_label,
        "punch_preserved": punch,
        "prefer_cleaner_quieter_when_target_met": prefer_cleaner,
        "handoff_mode": handoff_mode,
        "reference_lufs": round(float(ref_lufs), 3) if ref_lufs is not None else None,
        "candidate_lufs": round(float(cand_lufs), 3) if cand_lufs is not None else None,
        "commercial_floor_lufs": round(float(floor_lufs), 3),
        "reference_target_met": ref_met,
        "candidate_target_met": cand_met,
        "crest_loss_db": round(float(crest_loss), 3) if crest_loss is not None else None,
        "max_crest_loss_db": round(float(max_loss), 3) if max_loss is not None else None,
        "near_crest_loss_limit": near,
        "over_crest_loss_limit": over,
        "accepted": not bool(reject_cleaner or (over and punch and ref_met and handoff_mode not in {"commercial_density_completion", "guarded_completion"})),
        "reject_for_cleaner_quieter": bool(reject_cleaner),
        "action": "keep_cleaner_quieter_reference" if reject_cleaner else ("soften_future_chase" if near else "allow"),
        "policy": "BAMix v63.1.9 treats crest budget as hard cleaner/quieter preference only in transparent/fragile handoff; density-completion relies on existing damage gates for final veto.",
    }


def _v6314_2_bamix_loudness_target_guard(decision: dict[str, Any] | None, target_lufs: float, mode: str = "") -> dict[str, Any]:
    """Lower the main render target for BAMix punch-preserved handoff.

    v63.1.4 limited the commercial finish retry, but the standard render could
    already meet the floor while spending too much crest.  This guard acts at
    the actual _render_once loudness target so the first standard final chases
    the acceptance floor, not the louder preferred target, when BAMix asked to
    preserve punch.
    """
    hp = _v6314_handoff_policy_from_decision(decision)
    try:
        original = float(target_lufs)
    except Exception:
        original = -11.5
    if not hp.get("active") or not _env_bool("BUSY_BAMIX_V6314_2_TARGET_GUARD", True):
        return {
            "schema_version": "busy_v6314_2_bamix_loudness_target_guard",
            "active": False,
            "original_target_lufs": round(float(original), 3),
            "effective_target_lufs": round(float(original), 3),
            "reason": "inactive_or_not_bamix_handoff",
        }
    handoff_mode = _bamix_v6319_handoff_mode(decision)
    if handoff_mode in {"commercial_density_completion", "guarded_completion"}:
        return {
            "schema_version": "busy_v6314_2_bamix_loudness_target_guard",
            "active": False,
            "original_target_lufs": round(float(original), 3),
            "effective_target_lufs": round(float(original), 3),
            "handoff_mode": handoff_mode,
            "reason": "v6319_density_completion_handoff_does_not_floor_chase",
            "policy": "BAMix density-completion mode lets existing damage gates evaluate commercial push instead of preemptively reducing the render target to the floor.",
        }
    punch = bool(hp.get("punch_preserved") or str(hp.get("selected_mix_recipe") or os.environ.get("BUSY_BAMIX_SELECTED_RECIPE", "")).strip().lower() == "punch_preserved")
    prefer_cleaner = bool(hp.get("prefer_cleaner_quieter_when_target_met", True))
    if not (punch and prefer_cleaner):
        return {
            "schema_version": "busy_v6314_2_bamix_loudness_target_guard",
            "active": False,
            "original_target_lufs": round(float(original), 3),
            "effective_target_lufs": round(float(original), 3),
            "punch_preserved": punch,
            "prefer_cleaner_quieter_when_target_met": prefer_cleaner,
            "handoff_mode": handoff_mode,
            "reason": "recipe_or_policy_does_not_request_floor_chase",
        }
    floor = _airqc_num(hp.get("effective_floor_lufs"), None)
    if floor is None:
        floor = _airqc_num(os.environ.get("BUSY_AUTOMIX_MASTERING_EFFECTIVE_FLOOR_LUFS"), None)
    if floor is None:
        return {
            "schema_version": "busy_v6314_2_bamix_loudness_target_guard",
            "active": False,
            "original_target_lufs": round(float(original), 3),
            "effective_target_lufs": round(float(original), 3),
            "reason": "missing_effective_floor",
        }
    margin = _env_float("BUSY_BAMIX_V6314_2_FLOOR_MARGIN_DB", 0.18, 0.0, 0.55)
    floor_target = float(floor) + float(margin)
    # Negative LUFS: the numerically smaller value is quieter.
    effective = min(float(original), float(floor_target))
    changed = bool(effective < original - 1e-6)
    return {
        "schema_version": "busy_v6314_2_bamix_loudness_target_guard",
        "active": True,
        "applied": changed,
        "punch_preserved": punch,
        "prefer_cleaner_quieter_when_target_met": prefer_cleaner,
        "original_target_lufs": round(float(original), 3),
        "effective_floor_lufs": round(float(floor), 3),
        "floor_margin_db": round(float(margin), 3),
        "floor_chase_target_lufs": round(float(floor_target), 3),
        "effective_target_lufs": round(float(effective), 3),
        "target_reduction_db": round(float(original - effective), 3),
        "reason": "bamix_punch_preserved_floor_chase_to_reduce_standard_final_crest_loss" if changed else "target_already_at_or_below_floor_chase",
        "policy": "For BAMix punch-preserved premasters, the standard render should meet the commercial floor cleanly instead of chasing the louder preferred target before crest-loss QC can react.",
    }


def _commercial_loudness_finish_profile(decision: dict[str, Any], mode: str) -> dict[str, Any]:
    """DB-driven commercial loudness target/floor policy.

    v8.5.3.8-db keeps the repair-then-push finish pass, but derives its target
    and floor from the already-loaded private Supabase reference DB overlay
    (decision.reference_v7.mode_profile.integrated_lufs_range) whenever possible.
    Hard-coded family LUFS floors are intentionally avoided; family text is used
    only for non-target safety defaults such as crest tolerance and max push.
    """
    if not _env_bool("BUSY_COMMERCIAL_LOUDNESS_FINISH", True):
        return {"active": False, "reason": "disabled"}
    mode_l = str(mode or "").lower()
    hot_exp = _experimental_hot_commercial_enabled(mode)
    if ("streaming" in mode_l or "safe" in mode_l or "clean" in mode_l) and not _env_bool("BUSY_COMMERCIAL_FINISH_IN_SAFE_MODES", False):
        return {"active": False, "reason": "safe_or_clean_mode"}

    txt = _commercial_family_text(decision)
    tolerance_policy = _commercial_tolerance_matrix(txt)
    intensity_preset = str(tolerance_policy.get("intensity_preset") or _mastering_intensity_key())
    if intensity_preset == "safe" and _env_bool("BUSY_SAFE_DISABLE_COMMERCIAL_FINISH", True):
        return {
            "active": False,
            "reason": "safe_intensity_commercial_finish_disabled",
            "mastering_intensity": intensity_preset,
            "commercial_tolerance_policy": tolerance_policy,
            "busy_auto_mixing_mastering_profile": {"active": False, "reason": "commercial_finish_disabled_in_safe_intensity"},
            "schema_version": "busy_commercial_loudness_finish_v8_5_3_34_mode_aware_render_budget",
            "policy": "Safe intensity uses the composed standard chain and final safety ceiling; commercial target-chase candidates are not generated before/after Safe lock",
        }
    rv7 = decision.get("reference_v7", {}) if isinstance(decision.get("reference_v7", {}), dict) else {}
    targets = decision.get("targets", {}) if isinstance(decision.get("targets", {}), dict) else {}
    source_candidates = _commercial_loudness_range_sources(decision)
    primary_source = source_candidates[0] if source_candidates else {}
    pair = tuple(primary_source.get("range") or ()) if primary_source else None

    # Default fallback is derived from current decision targets, not from a fixed
    # family table.  It only runs when the private DB overlay is missing.
    if pair and len(pair) == 2:
        quiet_edge = float(pair[0])
        loud_edge = float(pair[1])
        target_position = _env_float("BUSY_COMMERCIAL_FINISH_DB_TARGET_POSITION", 0.35, 0.0, 1.0)
        floor_tolerance_db = _env_float("BUSY_COMMERCIAL_FINISH_DB_FLOOR_TOLERANCE_DB", 0.65, 0.0, 2.5)
        target_lufs = quiet_edge + (loud_edge - quiet_edge) * target_position
        floor_lufs = quiet_edge - floor_tolerance_db
        source = str(primary_source.get("source") or "unknown_lufs_range")
        if source.startswith("detailed_genre_decision"):
            try:
                safer_floor = float(primary_source.get("safer_floor_lufs"))
                # Use the research/reference safer commercial floor as the lower
                # acceptance boundary when present, but never let it exceed target.
                if -30.0 <= safer_floor <= -3.0:
                    floor_lufs = max(float(floor_lufs), min(safer_floor, target_lufs - 0.35))
            except Exception:
                pass
    else:
        fallback_target = targets.get("integrated_lufs_target")
        try:
            target_lufs = float(fallback_target)
        except Exception:
            fallback_pair = _lufs_range_to_pair(targets.get("integrated_lufs_range"))
            if fallback_pair is not None:
                target_lufs = sum(fallback_pair) / 2.0
            else:
                target_lufs = -9.25
        floor_lufs = target_lufs - _env_float("BUSY_COMMERCIAL_FINISH_FALLBACK_FLOOR_GAP_DB", 0.55, 0.25, 2.0)
        quiet_edge = floor_lufs
        loud_edge = target_lufs
        source = "decision.targets.integrated_lufs_target_fallback"

    family = str(rv7.get("family") or (decision.get("genre_chain", {}) or {}).get("selected_family") or "db_driven_commercial").strip() or "db_driven_commercial"
    crest_min = float(targets.get("crest_floor_db") or 5.7)
    hard_crest_min = max(4.4, crest_min - 0.55)
    max_push_db = 2.05
    harsh_hard_max_db = float(targets.get("harshness_delta_max_db") or 1.25) + 0.18
    allow_moderate_lra_warning = False

    # Non-target safety defaults still need genre/form awareness. These do not
    # create LUFS floors; they only shape how hard a retry candidate is allowed
    # to push before QC rejects it.
    if any(k in txt for k in ["hard_techno", "schranz", "hard_dance", "industrial_techno"]):
        max_push_db = 2.65
        crest_min = min(crest_min, 5.15)
        hard_crest_min = min(hard_crest_min, 4.70)
        harsh_hard_max_db = min(harsh_hard_max_db, 1.25)
    elif any(k in txt for k in ["edm", "electro_house", "house", "festival", "club", "bounce", "future_garage", "garage"]):
        max_push_db = 2.55
        crest_min = min(crest_min, 5.45)
        hard_crest_min = min(hard_crest_min, 4.95)
        harsh_hard_max_db = min(harsh_hard_max_db, 1.32)
    elif any(k in txt for k in ["kpop", "k_pop", "commercial_pop", "hyperpop", "dance_pop"]):
        max_push_db = 2.40
        crest_min = min(max(crest_min, 5.25), 5.75)
        hard_crest_min = min(hard_crest_min, 5.05)
        harsh_hard_max_db = min(harsh_hard_max_db, 1.30)
    elif any(k in txt for k in ["rock", "pop_rock", "band", "guitar"]):
        max_push_db = 1.95
        crest_min = max(crest_min, 5.85)
        hard_crest_min = max(hard_crest_min, 5.25)
        harsh_hard_max_db = min(harsh_hard_max_db, 1.25)
    elif any(k in txt for k in ["ballad", "acoustic", "live", "orchestral", "cinematic", "worship"]):
        max_push_db = 1.35
        crest_min = max(crest_min, 6.35)
        hard_crest_min = max(hard_crest_min, 5.70)
        harsh_hard_max_db = min(harsh_hard_max_db, 1.15)
        allow_moderate_lra_warning = True

    mode_profile = rv7.get("mode_profile", {}) if isinstance(rv7.get("mode_profile", {}), dict) else {}
    try:
        aggression = float(mode_profile.get("aggression_scalar") or 1.0)
    except Exception:
        aggression = 1.0
    max_push_db += max(-0.25, min(0.40, (aggression - 1.0) * 0.45))

    if "hot" in mode_l:
        target_lufs += 0.20
        floor_lufs += 0.18
        max_push_db += 0.30
    elif "auto commercial" in mode_l or mode_l.strip() == "auto":
        target_lufs += 0.02
        floor_lufs += 0.02

    if hot_exp:
        # User-reviewed experimental path: test genuinely hot commercial candidates.
        # Negative LUFS values get louder as they move toward zero, so max() chooses
        # the hotter target/floor while still allowing explicit env overrides below.
        exp_target = _env_float("BUSY_EXPERIMENTAL_HOT_TARGET_LUFS", -7.8, -10.5, -5.8)
        exp_floor_gap = _env_float("BUSY_EXPERIMENTAL_HOT_FLOOR_GAP_DB", 1.05, 0.35, 2.2)
        target_lufs = max(float(target_lufs), float(exp_target))
        floor_lufs = max(float(floor_lufs), float(target_lufs) - float(exp_floor_gap))
        max_push_db = max(float(max_push_db), _env_float("BUSY_EXPERIMENTAL_HOT_MAX_PUSH_DB", 4.2, 1.0, 6.0))
        crest_min = min(float(crest_min), _env_float("BUSY_EXPERIMENTAL_HOT_CREST_MIN_DB", 4.95, 4.0, 7.5))
        hard_crest_min = min(float(hard_crest_min), _env_float("BUSY_EXPERIMENTAL_HOT_HARD_CREST_MIN_DB", 4.45, 3.8, 7.0))
        # v8.5.3.20: for club/electronic hot-commercial candidates, crest
        # collapse is a blocker only after it falls below a true failure floor.
        # A crest around 4.4-5.0 dB can be normal for dense EDM/club masters and
        # should be user/QC-visible, not automatically rejected before audition.
        if any(k in txt for k in ["edm", "electro", "house", "club", "festival", "bounce", "hard_techno", "schranz"]):
            crest_min = min(float(crest_min), 4.75)
            hard_crest_min = min(float(hard_crest_min), 4.30)
        harsh_hard_max_db = max(float(harsh_hard_max_db), _env_float("BUSY_EXPERIMENTAL_HOT_HARSH_MAX_DB", 1.7, 0.8, 3.2))
        allow_moderate_lra_warning = True

    target_lufs += _env_float("BUSY_COMMERCIAL_FINISH_TARGET_OFFSET_DB", 0.0, -1.5, 1.5)
    floor_lufs += _env_float("BUSY_COMMERCIAL_FINISH_FLOOR_OFFSET_DB", 0.0, -1.5, 1.5)
    if hot_exp and _env_bool("BUSY_COMMERCIAL_DB_TARGET_CHASE", True):
        max_push_db = max(float(max_push_db), _env_float("BUSY_COMMERCIAL_DB_TARGET_MAX_PUSH_DB", 5.6, 2.0, 7.5))
    max_push_db = _env_float("BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB", max_push_db, 0.0, 7.5 if hot_exp else 3.2)
    crest_min = _env_float("BUSY_COMMERCIAL_FINISH_CREST_MIN_DB", crest_min, 4.2, 9.0)
    hard_crest_min = min(crest_min - 0.25, _env_float("BUSY_COMMERCIAL_FINISH_HARD_CREST_MIN_DB", hard_crest_min, 2.5, 8.5))

    # v8.5.3.31: auto-selected mastering intensity turns commercial
    # loudness degradation into a tolerance matrix.  Crest/LRA/transient
    # reductions can become score penalties in Commercial/Maximum, while
    # true peak, phase, clipping, vocal and low-end collapse stay hard vetoes.
    if intensity_preset in {"commercial", "maximum"}:
        crest_min = min(float(crest_min), float(tolerance_policy.get("crest_warning_min_db", crest_min)))
        hard_crest_min = min(float(hard_crest_min), float(tolerance_policy.get("crest_hard_min_db", hard_crest_min)))
        allow_moderate_lra_warning = True
    elif intensity_preset == "balanced":
        crest_min = min(float(crest_min), float(tolerance_policy.get("crest_warning_min_db", crest_min)))
        hard_crest_min = min(float(hard_crest_min), float(tolerance_policy.get("crest_hard_min_db", hard_crest_min)))
    else:
        crest_min = max(float(crest_min), float(tolerance_policy.get("crest_warning_min_db", crest_min)))
        hard_crest_min = max(float(hard_crest_min), float(tolerance_policy.get("crest_hard_min_db", hard_crest_min)))

    residue_ctx = {}
    try:
        rtc = decision.get("runtime_safety_context", {}) if isinstance(decision.get("runtime_safety_context", {}), dict) else {}
        residue_ctx = rtc.get("frequency_band_centrifuge", {}) if isinstance(rtc.get("frequency_band_centrifuge", {}), dict) else {}
        if residue_ctx.get("active") and _env_bool("BUSY_RESIDUE_INDIRECT_INFLUENCE", True):
            rr = float(residue_ctx.get("risk_score", 0.0) or 0.0)
            hf_scalar = float(residue_ctx.get("limiter_hf_push_scalar", 1.0) or 1.0)
            if hot_exp:
                # Experimental mode lets direct light control handle the residue
                # blocker, so do not hard-cap loudness to a quiet master.
                max_push_db *= float(np.clip(hf_scalar, 0.86, 1.0))
                if rr >= 0.82:
                    max_push_db = min(max_push_db, 3.4)
                harsh_hard_max_db = max(harsh_hard_max_db, 1.45)
            else:
                # Do not turn residue risk into a quiet master by default, but make
                # the final repair-then-push less aggressive until direct cleanup is trusted.
                max_push_db *= float(np.clip(hf_scalar, 0.62, 1.0))
                if rr >= 0.78:
                    max_push_db = min(max_push_db, 1.35)
                elif rr >= 0.62:
                    max_push_db = min(max_push_db, 1.75)
                harsh_hard_max_db = min(harsh_hard_max_db, 1.10 if rr >= 0.70 else harsh_hard_max_db)
    except Exception:
        residue_ctx = {}

    bamix_policy = _bamix_handoff_policy_from_decision(decision)
    if bamix_policy.get("active") and _env_bool("BUSY_AUTOMIX_MASTERING_PROFILE_ENABLE", True):
        handoff_mode = _bamix_v6319_handoff_mode(decision)
        bamix_target = _bamix_policy_float(bamix_policy, "effective_final_target_lufs", "BUSY_AUTOMIX_MASTERING_EFFECTIVE_TARGET_LUFS", -11.5, -14.0, -7.8)
        bamix_floor = _bamix_policy_float(bamix_policy, "effective_floor_lufs", "BUSY_AUTOMIX_MASTERING_EFFECTIVE_FLOOR_LUFS", bamix_target - 0.7, -15.5, -7.9)
        bamix_max_push = _bamix_policy_float(bamix_policy, "max_finish_push_db", "BUSY_AUTOMIX_MASTERING_FINISH_MAX_PUSH_DB", 2.0, 0.0, 5.0)
        if handoff_mode in {"commercial_density_completion", "guarded_completion"}:
            # v63.1.9: let commercial finish complete density when BAMix is
            # structurally healthy but under-commercial. Existing candidate
            # gates still decide whether the louder result is acceptable.
            target_lufs = float(bamix_target)
            if _env_bool("BUSY_BAMIX_V6319_ALLOW_DB_HOTTER_THAN_POLICY", False):
                target_lufs = max(float(target_lufs), float(bamix_policy.get("db_preferred_target_lufs") or target_lufs))
            hottest = _env_float("BUSY_BAMIX_V6319_HOTTEST_EFFECTIVE_TARGET_LUFS", -8.8, -10.5, -7.8)
            if float(target_lufs) > float(hottest):
                target_lufs = float(hottest)
            floor_lufs = float(bamix_floor)
            max_push_db = min(max(float(max_push_db), min(float(bamix_max_push), 3.2 if handoff_mode == "commercial_density_completion" else 2.35)), float(bamix_max_push))
            crest_min = min(float(crest_min), _env_float("BUSY_BAMIX_V6319_DENSITY_CREST_WARNING_MIN_DB", 7.2 if handoff_mode == "commercial_density_completion" else 7.8, 4.5, 10.0))
            hard_crest_min = min(float(hard_crest_min), _env_float("BUSY_BAMIX_V6319_DENSITY_HARD_CREST_MIN_DB", 6.6 if handoff_mode == "commercial_density_completion" else 7.1, 4.0, 9.5))
            allow_moderate_lra_warning = True
        else:
            bamix_crest_floor = _env_float("BUSY_AUTOMIX_MASTERING_CREST_MIN_DB", 8.8, 6.0, 12.0)
            bamix_hard_crest_floor = _env_float("BUSY_AUTOMIX_MASTERING_HARD_CREST_MIN_DB", 8.3, 5.5, 11.5)
            target_lufs = min(float(target_lufs), float(bamix_target))
            floor_lufs = min(float(floor_lufs), float(bamix_floor), float(target_lufs) - 0.35)
            max_push_db = min(float(max_push_db), float(bamix_max_push))
            crest_min = max(float(crest_min), float(bamix_crest_floor))
            hard_crest_min = max(float(hard_crest_min), float(bamix_hard_crest_floor))
            hot_exp = False
            allow_moderate_lra_warning = False
        bamix_profile_override = {
            "active": True,
            "schema_version": "busy_auto_mixing_commercial_finish_profile_v8_5_3_63_1_9",
            "profile": bamix_policy.get("profile") or ("bamix_density_completion" if handoff_mode in {"commercial_density_completion", "guarded_completion"} else "transparent_bamix_premaster_mastering"),
            "handoff_mode": handoff_mode,
            "commercial_readiness": bamix_policy.get("commercial_readiness", {}),
            "target_lufs": round(float(target_lufs), 3),
            "commercial_floor_lufs": round(float(floor_lufs), 3),
            "max_push_db": round(float(max_push_db), 3),
            "crest_min_db": round(float(crest_min), 3),
            "hard_crest_min_db": round(float(hard_crest_min), 3),
            "premaster_lra_lu": bamix_policy.get("premaster_lra_lu"),
            "premaster_crest_factor_db": bamix_policy.get("premaster_crest_factor_db"),
            "selected_mix_recipe": bamix_policy.get("selected_mix_recipe"),
            "punch_preserved": bool(bamix_policy.get("punch_preserved")),
            "prefer_cleaner_quieter_when_target_met": bool(bamix_policy.get("prefer_cleaner_quieter_when_target_met", True)),
            "adaptive_damage_governor": bamix_policy.get("adaptive_damage_governor", {}),
            "source_policy": bamix_policy,
            "policy": "BAMix v63.1.9 uses readiness-aware handoff: transparent when ready/fragile, density completion when under-commercial but safe, with existing damage gates as final authority.",
        }
    else:
        bamix_profile_override = {"active": False}

    # Keep the floor below the target; target/floor are negative LUFS values.
    if floor_lufs > target_lufs - 0.25:
        floor_lufs = target_lufs - 0.45
    # Avoid accepting obviously quiet results when the private DB asks for a hot
    # master, but do not make a missing DB fallback dangerously aggressive.
    if source.startswith("reference_v7"):
        floor_lufs = max(floor_lufs, target_lufs - _env_float("BUSY_COMMERCIAL_FINISH_MAX_TARGET_FLOOR_GAP_DB", 1.45, 0.45, 3.0))

    return {
        "active": True,
        "schema_version": "busy_commercial_loudness_finish_v8_5_3_31_auto_intensity_tolerance_matrix" if hot_exp else "busy_commercial_loudness_finish_v8_5_3_31_profile_stack_db_driven",
        "mastering_intensity": intensity_preset,
        "commercial_tolerance_policy": tolerance_policy,
        "busy_auto_mixing_mastering_profile": bamix_profile_override,
        "experimental_hot_commercial": bool(hot_exp),
        "adaptive_hot_commercial": bool(hot_exp),
        "commercial_reference_match": bool(_env_bool("BUSY_COMMERCIAL_DB_TARGET_CHASE", True) and hot_exp),
        "reference_true_peak_ceiling_db": round(float(COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB), 3),
        "family": family,
        "target_lufs": round(float(target_lufs), 3),
        "commercial_floor_lufs": round(float(floor_lufs), 3),
        "db_lufs_range_source": source,
        "db_integrated_lufs_range": [round(float(quiet_edge), 3), round(float(loud_edge), 3)],
        "available_lufs_range_sources": source_candidates,
        "crest_min_db": round(float(crest_min), 3),
        "hard_crest_min_db": round(float(hard_crest_min), 3),
        "max_push_db": round(float(max_push_db), 3),
        "harshness_hard_max_db": round(float(harsh_hard_max_db), 3),
        "allow_moderate_lra_warning": bool(allow_moderate_lra_warning),
        "frequency_band_centrifuge_influence": residue_ctx if isinstance(residue_ctx, dict) and residue_ctx.get("active") else {"active": False},
        "policy": "adaptive hot-commercial mode: measured residue/bloom blockers are corrected first, then louder candidates are tested and scored; user judges by ear while stage_state records the data" if hot_exp else "private_reference_db_lufs_range_drives_target_and_floor; repair_control_then_push; loudness rollback is the last resort; residue centrifuge may cap high-band-exposing push indirectly",
    }

def _analysis_summary_for_loudness(analysis: dict[str, Any] | None) -> dict[str, Any]:
    analysis = analysis or {}
    ms = analysis.get("ms", {}) if isinstance(analysis.get("ms", {}), dict) else {}
    q = analysis.get("quality_indices", {}) if isinstance(analysis.get("quality_indices", {}), dict) else {}
    measurement = analysis.get("measurement_authority") if isinstance(analysis.get("measurement_authority"), dict) else (analysis.get("loudness", {}) or {}).get("measurement_authority") if isinstance(analysis.get("loudness", {}), dict) else {}
    return {
        "integrated_lufs": analysis.get("integrated_lufs"),
        "stereo_integrated_lufs": analysis.get("stereo_integrated_lufs") or (analysis.get("loudness", {}) or {}).get("stereo_integrated_lufs") if isinstance(analysis.get("loudness", {}), dict) else analysis.get("stereo_integrated_lufs"),
        "mono_downmix_lufs": analysis.get("mono_downmix_lufs") or (analysis.get("loudness", {}) or {}).get("mono_downmix_lufs") if isinstance(analysis.get("loudness", {}), dict) else analysis.get("mono_downmix_lufs"),
        "lufs_authority": analysis.get("lufs_authority") or (analysis.get("loudness", {}) or {}).get("lufs_authority") if isinstance(analysis.get("loudness", {}), dict) else analysis.get("lufs_authority"),
        "measurement_authority": measurement,
        "true_peak_dbtp": analysis.get("approx_true_peak_dbfs"),
        "crest_db": analysis.get("crest_factor_db"),
        "correlation": analysis.get("correlation"),
        "side_low_ratio": ms.get("side_ratio_lowband"),
        "side_high_ratio": ms.get("side_ratio_8k_14k"),
        "harshness_index_db": q.get("harshness_index_db"),
        "air_index_db": q.get("air_index_db"),
        "lra_lu": _analysis_lra_lu(analysis),
    }


def _v62_8_1_metric_from_analysis(analysis: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    analysis = analysis if isinstance(analysis, dict) else {}
    for key in keys:
        try:
            val = analysis.get(key)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    loud = analysis.get("loudness") if isinstance(analysis.get("loudness"), dict) else {}
    for key in keys:
        try:
            val = loud.get(key)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    meas = analysis.get("measurement_authority") if isinstance(analysis.get("measurement_authority"), dict) else loud.get("measurement_authority") if isinstance(loud.get("measurement_authority"), dict) else {}
    for key in keys:
        try:
            val = meas.get(key)
            if val is not None and np.isfinite(float(val)):
                return float(val)
        except Exception:
            pass
    return default


def _v62_8_1_apply_measurement_authority_to_finish(finish: dict[str, Any] | None, final_analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Make final commercial-floor/target decisions use full stereo analysis.

    Earlier measurement paths could let mono-downmix LUFS leak into final_lufs or
    commercial_floor decisions.  This post-final-analysis pass is intentionally
    small and deterministic: it does not create DSP or loudness push; it only
    corrects the reported and learned outcome fields using the authoritative
    full-render analysis.
    """
    if not isinstance(finish, dict):
        return finish or {}
    fa = final_analysis if isinstance(final_analysis, dict) else {}
    stereo_lufs = _v62_8_1_metric_from_analysis(fa, "stereo_integrated_lufs", "integrated_lufs", default=None)
    mono_lufs = _v62_8_1_metric_from_analysis(fa, "mono_downmix_lufs", default=None)
    integrated_lufs = _v62_8_1_metric_from_analysis(fa, "integrated_lufs", default=stereo_lufs)
    target = _v62_8_num(finish.get("target_lufs"), None)
    floor = _v62_8_num(finish.get("commercial_floor_lufs"), None)
    if stereo_lufs is None:
        return finish
    authority = {
        "schema_version": "busy_v62_8_1_final_measurement_authority",
        "active": True,
        "authority": "stereo_integrated_lufs",
        "final_integrated_lufs": round(float(stereo_lufs), 3),
        "final_stereo_lufs": round(float(stereo_lufs), 3),
        "final_mono_downmix_lufs": round(float(mono_lufs), 3) if mono_lufs is not None else None,
        "previous_finish_target_met": finish.get("target_met"),
        "previous_finish_target_reached": finish.get("target_reached"),
        "previous_finish_input_lufs": finish.get("input", {}).get("integrated_lufs") if isinstance(finish.get("input"), dict) else None,
        "final_analysis_integrated_lufs": round(float(integrated_lufs), 3) if integrated_lufs is not None else None,
        "stereo_minus_mono_lufs_db": round(float(stereo_lufs - mono_lufs), 3) if mono_lufs is not None else None,
        "warning": "mono_downmix_lufs_differs_from_stereo_authority" if mono_lufs is not None and abs(float(stereo_lufs - mono_lufs)) >= _env_float("BUSY_MEASUREMENT_MONO_STEREO_LUFS_WARN_DB", 1.5, 0.0, 12.0) else None,
        "policy": "Final commercial floor/target outcome is decided from full-render stereo integrated LUFS; mono downmix LUFS is diagnostic only.",
    }
    if floor is not None:
        finish["target_met_before_measurement_authority"] = finish.get("target_met")
        finish["target_met"] = bool(float(stereo_lufs) >= float(floor))
        authority["commercial_floor_lufs"] = round(float(floor), 3)
        authority["commercial_floor_met"] = bool(finish["target_met"])
    if target is not None:
        finish["target_reached_before_measurement_authority"] = finish.get("target_reached")
        finish["target_reached"] = bool(float(stereo_lufs) >= float(target) - 0.20)
        authority["target_lufs"] = round(float(target), 3)
        authority["target_reached"] = bool(finish["target_reached"])
    finish["final_measurement_authority"] = authority
    finish["final_lufs_authority"] = "stereo_integrated_lufs"
    finish["final_lufs_stereo"] = round(float(stereo_lufs), 3)
    finish["final_lufs_mono_downmix"] = round(float(mono_lufs), 3) if mono_lufs is not None else None
    return finish


def _v62_8_2_apply_measurement_authority_runtime_safe(finish: dict[str, Any] | None, final_analysis: dict[str, Any] | None) -> dict[str, Any]:
    """Small non-recursive final LUFS authority update for runtime-hotfix paths."""
    if not isinstance(finish, dict):
        return finish or {}
    fa = final_analysis if isinstance(final_analysis, dict) else {}
    loud = fa.get("loudness") if isinstance(fa.get("loudness"), dict) else {}
    meas = fa.get("measurement_authority") if isinstance(fa.get("measurement_authority"), dict) else loud.get("measurement_authority") if isinstance(loud.get("measurement_authority"), dict) else {}
    def _f(v):
        try:
            x = float(v)
            return x if np.isfinite(x) else None
        except Exception:
            return None
    stereo = _f(fa.get("stereo_integrated_lufs"))
    if stereo is None:
        stereo = _f(loud.get("stereo_integrated_lufs"))
    if stereo is None:
        stereo = _f(meas.get("stereo_integrated_lufs"))
    if stereo is None:
        stereo = _f(fa.get("integrated_lufs"))
    mono = _f(fa.get("mono_downmix_lufs"))
    if mono is None:
        mono = _f(loud.get("mono_downmix_lufs"))
    if mono is None:
        mono = _f(meas.get("mono_downmix_lufs"))
    target = _f(finish.get("target_lufs"))
    floor = _f(finish.get("commercial_floor_lufs"))
    if stereo is None:
        finish.setdefault("final_measurement_authority", {"schema_version": "busy_v62_9_3_final_measurement_authority_runtime_safe", "active": False, "reason": "missing_stereo_lufs"})
        return finish
    authority = {
        "schema_version": "busy_v62_9_3_final_measurement_authority_runtime_safe",
        "active": True,
        "authority": "stereo_integrated_lufs",
        "final_integrated_lufs": round(float(stereo), 3),
        "final_stereo_lufs": round(float(stereo), 3),
        "final_mono_downmix_lufs": round(float(mono), 3) if mono is not None else None,
        "previous_finish_target_met": finish.get("target_met"),
        "previous_finish_target_reached": finish.get("target_reached"),
        "stereo_minus_mono_lufs_db": round(float(stereo - mono), 3) if mono is not None else None,
        "policy": "Runtime-safe final commercial floor/target outcome is decided from final stereo integrated LUFS; no DSP is created here.",
    }
    if floor is not None:
        finish["target_met_before_measurement_authority"] = finish.get("target_met")
        finish["target_met"] = bool(float(stereo) >= float(floor))
        authority["commercial_floor_lufs"] = round(float(floor), 3)
        authority["commercial_floor_met"] = bool(finish["target_met"])
    if target is not None:
        finish["target_reached_before_measurement_authority"] = finish.get("target_reached")
        finish["target_reached"] = bool(float(stereo) >= float(target) - 0.20)
        authority["target_lufs"] = round(float(target), 3)
        authority["target_reached"] = bool(finish["target_reached"])
    finish["final_measurement_authority"] = authority
    finish["final_lufs_authority"] = "stereo_integrated_lufs"
    finish["final_lufs_stereo"] = round(float(stereo), 3)
    finish["final_lufs_mono_downmix"] = round(float(mono), 3) if mono is not None else None
    return finish



def _v62_9_4_1_attach_top_level_final_metric_aliases(report: dict[str, Any] | None, final_analysis: dict[str, Any] | None, finish: dict[str, Any] | None) -> dict[str, Any]:
    """v62.9.4.1: keep final metric aliases visible after stem-policy hotfixes.

    v62.9.3 made stereo LUFS authority available to downstream/debug readers.
    v62.9.4 changed Stem Gate semantics and touched the final report paths; this
    helper prevents the authoritative metrics from remaining only in nested
    output_analysis/commercial_loudness_finish fields.  It is telemetry-only and
    never changes audio or candidate selection.
    """
    if not isinstance(report, dict):
        report = {}
    fa = final_analysis if isinstance(final_analysis, dict) else {}
    loud = fa.get("loudness") if isinstance(fa.get("loudness"), dict) else {}
    meas = fa.get("measurement_authority") if isinstance(fa.get("measurement_authority"), dict) else loud.get("measurement_authority") if isinstance(loud.get("measurement_authority"), dict) else {}
    fin = finish if isinstance(finish, dict) else {}
    auth = fin.get("final_measurement_authority") if isinstance(fin.get("final_measurement_authority"), dict) else {}

    def _finite(v: Any) -> float | None:
        try:
            x = float(v)
            return x if np.isfinite(x) else None
        except Exception:
            return None

    def _first(*vals: Any) -> float | None:
        for v in vals:
            x = _finite(v)
            if x is not None:
                return x
        return None

    stereo = _first(fin.get("final_lufs_stereo"), auth.get("final_stereo_lufs"), fa.get("stereo_integrated_lufs"), loud.get("stereo_integrated_lufs"), meas.get("stereo_integrated_lufs"), fa.get("integrated_lufs"), loud.get("integrated_lufs"))
    mono = _first(fin.get("final_lufs_mono_downmix"), auth.get("final_mono_downmix_lufs"), fa.get("mono_downmix_lufs"), loud.get("mono_downmix_lufs"), meas.get("mono_downmix_lufs"))
    lufs = _first(fa.get("integrated_lufs"), loud.get("integrated_lufs"), stereo)
    tp = _first(fa.get("approx_true_peak_dbfs"), loud.get("approx_true_peak_dbfs"), loud.get("true_peak_dbfs"), fa.get("true_peak_dbfs"))
    crest = _first(fa.get("crest_factor_db"), loud.get("crest_factor_db"))
    st = loud.get("short_term_lufs") if isinstance(loud.get("short_term_lufs"), dict) else {}
    lra = _first(fa.get("lra_lu"), st.get("lra_lu"))

    if lufs is not None:
        report["final_lufs"] = round(float(lufs), 3)
    report["final_lufs_authority"] = fin.get("final_lufs_authority") or auth.get("authority") or meas.get("authority") or fa.get("lufs_authority") or loud.get("lufs_authority")
    if stereo is not None:
        report["final_lufs_stereo"] = round(float(stereo), 3)
    if mono is not None:
        report["final_lufs_mono_downmix"] = round(float(mono), 3)
    if stereo is not None and mono is not None:
        report["stereo_minus_mono_lufs_db"] = round(float(stereo - mono), 3)
    if tp is not None:
        tp_r = round(float(tp), 3)
        report["true_peak_dbtp"] = tp_r
        # v62.9.9: keep all final true-peak aliases on the same authority.
        # Previous reports could show output_analysis.true_peak_dbtp from an older
        # alias while top-level/debug used approx_true_peak_dbfs, confusing QC.
        try:
            fa["true_peak_dbtp"] = tp_r
            fa["approx_true_peak_dbfs"] = tp_r
            fa["true_peak_dbfs"] = tp_r
            if isinstance(loud, dict):
                loud["true_peak_dbtp"] = tp_r
                loud["approx_true_peak_dbfs"] = tp_r
                loud["true_peak_dbfs"] = tp_r
        except Exception:
            pass
    if lra is not None:
        report["lra_lu"] = round(float(lra), 3)
    if crest is not None:
        report["crest_factor_db"] = round(float(crest), 3)
    report["target_lufs"] = fin.get("target_lufs")
    report["commercial_floor_lufs"] = fin.get("commercial_floor_lufs")
    report["target_met"] = fin.get("target_met")

    # v63.6.2.2: `target_met` is a legacy commercial-floor boolean in this
    # code path.  Surface exact target/floor semantics separately so DML target
    # miss policy is not confused with floor compliance.  Telemetry-only.
    _v6362_target = _first(fin.get("target_lufs"), report.get("target_lufs"), report.get("preferred_target_lufs"), report.get("actual_render_target_lufs"))
    _v6362_floor = _first(fin.get("commercial_floor_lufs"), report.get("commercial_floor_lufs"))
    _v6362_final = _first(report.get("final_lufs"), lufs, stereo)
    _v6362_contract = report.get("v6362_dml_ownership_contract") if isinstance(report.get("v6362_dml_ownership_contract"), dict) else {}
    _v6362_v60 = report.get("v60_deterministic_multistage_limiter") if isinstance(report.get("v60_deterministic_multistage_limiter"), dict) else {}
    if not _v6362_contract and isinstance(_v6362_v60.get("v6362_dml_ownership_contract"), dict):
        _v6362_contract = _v6362_v60.get("v6362_dml_ownership_contract") or {}
    _v6362_gate = _v6362_v60.get("acceptance_gate") if isinstance(_v6362_v60.get("acceptance_gate"), dict) else {}
    if not _v6362_contract and isinstance(_v6362_gate.get("v6362_dml_ownership_contract"), dict):
        _v6362_contract = _v6362_gate.get("v6362_dml_ownership_contract") or {}
    _v6362_floor_met = bool(_v6362_final is not None and _v6362_floor is not None and float(_v6362_final) >= float(_v6362_floor))
    _v6362_target_reached_calc = bool(_v6362_final is not None and _v6362_target is not None and float(_v6362_final) >= float(_v6362_target) - 0.20)
    # v63.6.2.4: target_reached is calculated from delivered final_lufs and target_lufs.
    # Do not trust commercial_finish.target_reached here; older branches can carry a
    # stale flag from a candidate/floor check rather than the delivered final master.
    _v6362_legacy_finish_target_reached = fin.get("target_reached")
    _v6362_target_reached = _v6362_target_reached_calc
    _v6362_target_gap = (float(_v6362_target) - float(_v6362_final)) if _v6362_target is not None and _v6362_final is not None else None
    _v6362_floor_margin = (float(_v6362_final) - float(_v6362_floor)) if _v6362_floor is not None and _v6362_final is not None else None
    _v6362_target_miss_allowed = _v6362_contract.get("target_miss_allowed") if isinstance(_v6362_contract, dict) else None
    report["commercial_floor_met"] = bool(_v6362_floor_met) if _v6362_floor is not None and _v6362_final is not None else report.get("commercial_floor_met")
    report["target_reached"] = bool(_v6362_target_reached) if _v6362_target is not None and _v6362_final is not None else report.get("target_reached")
    if _v6362_target_gap is not None:
        report["target_lufs_gap_lu"] = round(float(_v6362_target_gap), 3)
    if _v6362_floor_margin is not None:
        report["commercial_floor_margin_lu"] = round(float(_v6362_floor_margin), 3)
    if _v6362_target_miss_allowed is not None:
        _v6362_target_miss_bool = bool(_v6362_target_miss_allowed)
        report["target_miss_allowed"] = _v6362_target_miss_bool
        # v63.7.0.1: keep the explicit DML-named top-level alias that the
        # debug brief and deployment smoke checks expect.  This is telemetry
        # only; the DML contract remains the source of truth.
        report["target_miss_allowed_by_dml"] = _v6362_target_miss_bool
    report["target_semantics_v6362_2"] = {
        "schema_version": "busy_v6362_10_target_source_telemetry",
        "active": bool(_v6362_final is not None),
        "legacy_target_met_means_commercial_floor_met": True,
        "final_lufs": round(float(_v6362_final), 3) if _v6362_final is not None else None,
        "target_lufs": round(float(_v6362_target), 3) if _v6362_target is not None else None,
        "commercial_floor_lufs": round(float(_v6362_floor), 3) if _v6362_floor is not None else None,
        "commercial_floor_met": bool(_v6362_floor_met) if _v6362_floor is not None and _v6362_final is not None else None,
        "target_reached": bool(_v6362_target_reached) if _v6362_target is not None and _v6362_final is not None else None,
        "target_reach_tolerance_lu": 0.20,
        "target_reached_authority": "calculated_from_delivered_final_lufs_and_target_lufs_v6362_10",
        "legacy_finish_target_reached": bool(_v6362_legacy_finish_target_reached) if _v6362_legacy_finish_target_reached is not None else None,
        "target_lufs_gap_lu": round(float(_v6362_target_gap), 3) if _v6362_target_gap is not None else None,
        "commercial_floor_margin_lu": round(float(_v6362_floor_margin), 3) if _v6362_floor_margin is not None else None,
        "target_miss_allowed_by_dml_contract": bool(_v6362_target_miss_allowed) if _v6362_target_miss_allowed is not None else None,
        "dml_contract_passed": bool(_v6362_contract.get("contract_passed")) if isinstance(_v6362_contract, dict) and _v6362_contract.get("contract_passed") is not None else None,
        "policy": "Telemetry only: target_met remains the legacy commercial-floor boolean; exact target reach and DML-allowed target miss are surfaced separately.",
    }
    # v63.6.2.11: expose the current schema under current-version keys while
    # keeping older aliases for existing dashboards and Supabase queries.
    report["target_semantics_v6362_10"] = dict(report["target_semantics_v6362_2"])
    report["target_semantics_v6362_9"] = dict(report["target_semantics_v6362_2"])
    report["target_semantics_v6362_8"] = dict(report["target_semantics_v6362_2"])
    report["target_semantics_v6362_7"] = dict(report["target_semantics_v6362_2"])
    report["target_semantics_v6362_6"] = dict(report["target_semantics_v6362_2"])
    report["target_semantics_v6362_5"] = dict(report["target_semantics_v6362_2"])
    report["target_semantics_v6362_4"] = dict(report["target_semantics_v6362_2"])
    _v6362_clr = report.get("commercial_loudness_result") if isinstance(report.get("commercial_loudness_result"), dict) else {}
    if isinstance(_v6362_clr, dict):
        _v6362_clr["commercial_floor_met"] = report.get("commercial_floor_met")
        _v6362_clr["target_reached"] = report.get("target_reached")
        _v6362_clr["target_lufs_gap_lu"] = report.get("target_lufs_gap_lu")
        _v6362_clr["target_miss_allowed"] = report.get("target_miss_allowed")
        _v6362_clr["target_miss_allowed_by_dml"] = report.get("target_miss_allowed_by_dml", report.get("target_miss_allowed"))
        _v6362_clr["target_semantics_v6362_10"] = report["target_semantics_v6362_10"]
        _v6362_clr["target_semantics_v6362_9"] = report["target_semantics_v6362_9"]
        _v6362_clr["target_semantics_v6362_8"] = report["target_semantics_v6362_8"]
        _v6362_clr["target_semantics_v6362_7"] = report["target_semantics_v6362_7"]
        _v6362_clr["target_semantics_v6362_6"] = report["target_semantics_v6362_6"]
        _v6362_clr["target_semantics_v6362_2"] = report["target_semantics_v6362_2"]
        report["commercial_loudness_result"] = _v6362_clr

    # v63.6.2.4: preserve the already-authoritative top-level final_source,
    # but do not preserve a known stale legacy standard label when v60/DML is the
    # accepted/applied delivered source.  v63.6.2.3 still preserved an existing
    # top-level `standard_final_limiter_output` unconditionally; that could keep
    # a stale debug label if v63.6.2.2 had already polluted the top-level report.
    _v63623_prev_report_source = report.get("final_source")
    _v63623_finish_selected = fin.get("selected_final_source")
    _v63623_finish_fallback = fin.get("fallback_final_source")
    _v63623_report_selected = report.get("selected_final_source")
    _v63623_v60_applied = bool(isinstance(_v6362_v60, dict) and (_v6362_v60.get("applied") or (_v6362_v60.get("active") and _v6362_v60.get("accepted"))))
    _v63624_export_lock = report.get("final_export_lock") if isinstance(report.get("final_export_lock"), dict) else {}
    _v63624_rollback_to_standard = bool(
        _v63624_export_lock.get("rollback_to_standard_final")
        or _v63624_export_lock.get("rollback_triggered")
        or fin.get("rollback_triggered")
    )
    _v63624_stale_standard_labels = {
        "standard_final_limiter_output",
        "standard_final_limiter_output_after_v63162_floor_miss",
        "standard_final_after_v57_damage_gate",
        "standard_final_after_export_lock",
    }
    _v63625_v60_source_labels = {
        "v60_deterministic_multistage_limiter",
        "v60_deterministic_multistage_limiter_output",
        "dml_deterministic_multistage_limiter",
    }
    _v63624_prev_source_str = str(_v63623_prev_report_source or "")
    _v63624_prev_is_stale_standard = bool(
        _v63624_prev_source_str in _v63624_stale_standard_labels
        or (_v63624_prev_source_str.startswith("standard_final") and not _v63624_rollback_to_standard)
    )
    # v63.6.2.5: v63.6.2.4 correctly stopped preserving stale standard labels
    # when v60/DML was accepted, but the inverse rollback edge case still existed:
    # if an earlier telemetry pass had already written a v60 final_source and export
    # lock/damage guard later rolled back to the standard final, that stale v60 label
    # could be preserved.  Rollback-to-standard is an explicit delivery authority and
    # must win over any older v60/DML label.
    _v63625_prev_is_stale_v60_after_rollback = bool(
        _v63624_rollback_to_standard
        and (
            _v63624_prev_source_str in _v63625_v60_source_labels
            or _v63624_prev_source_str.startswith("v60_")
            or _v63624_prev_source_str.startswith("dml_")
        )
    )
    _v63625_standard_rollback_source = (
        _v63623_finish_fallback
        if isinstance(_v63623_finish_fallback, str) and _v63623_finish_fallback.startswith("standard_final")
        else (_v63623_finish_selected if isinstance(_v63623_finish_selected, str) and _v63623_finish_selected.startswith("standard_final") else None)
    ) or "standard_final_after_export_lock"
    _v63627_rollback_overrode_prior_final_source = bool(
        _v63624_rollback_to_standard
        and bool(_v63623_prev_report_source)
        and str(_v63623_prev_report_source or "") != str(_v63625_standard_rollback_source or "")
    )
    _v63623_resolution_reason = "fallback_standard_final_limiter_output"
    # v63.6.2.7: rollback-to-standard is an explicit delivered-source
    # authority.  v63.6.2.5 only overrode stale v60/DML labels after rollback,
    # but any pre-existing non-rollback final_source can be stale once export
    # lock/damage authority rolls back to the standard final.  Therefore the
    # rollback source must be resolved before preserving top-level final_source.
    if _v63624_rollback_to_standard:
        _v63623_final_source = _v63625_standard_rollback_source
        if _v63625_prev_is_stale_v60_after_rollback:
            _v63623_resolution_reason = "rollback_to_standard_overrode_stale_top_level_v60_source"
        elif _v63627_rollback_overrode_prior_final_source:
            _v63623_resolution_reason = "rollback_to_standard_overrode_prior_top_level_final_source"
        else:
            _v63623_resolution_reason = "derived_from_rollback_to_standard_authority"
    elif _v63623_v60_applied and _v63624_prev_is_stale_standard:
        _v63623_final_source = "v60_deterministic_multistage_limiter"
        _v63623_resolution_reason = "overrode_stale_top_level_standard_source_with_v60_applied_acceptance"
    elif _v63623_prev_report_source:
        _v63623_final_source = _v63623_prev_report_source
        _v63623_resolution_reason = "preserved_existing_top_level_report_final_source"
    elif _v63623_v60_applied:
        _v63623_final_source = "v60_deterministic_multistage_limiter"
        _v63623_resolution_reason = "derived_from_v60_applied_acceptance"
    elif _v63623_report_selected:
        _v63623_final_source = _v63623_report_selected
        _v63623_resolution_reason = "derived_from_report_selected_final_source"
    elif _v63623_finish_selected:
        _v63623_final_source = _v63623_finish_selected
        _v63623_resolution_reason = "derived_from_commercial_finish_selected_final_source"
    elif _v63623_finish_fallback:
        _v63623_final_source = _v63623_finish_fallback
        _v63623_resolution_reason = "derived_from_commercial_finish_fallback_final_source"
    else:
        _v63623_final_source = "standard_final_limiter_output"
    report["final_source"] = _v63623_final_source
    _v63624_final_source_authority = {
        "schema_version": "busy_v6362_10_final_source_authority",
        "active": True,
        "final_source": report.get("final_source"),
        "previous_report_final_source": _v63623_prev_report_source,
        "previous_report_final_source_was_stale_standard": bool(_v63624_prev_is_stale_standard),
        "previous_report_final_source_was_stale_v60_after_rollback": bool(_v63625_prev_is_stale_v60_after_rollback),
        "previous_report_final_source_overridden_by_rollback_standard": bool(_v63627_rollback_overrode_prior_final_source),
        "standard_rollback_source": _v63625_standard_rollback_source if _v63624_rollback_to_standard else None,
        "commercial_finish_selected_final_source": _v63623_finish_selected,
        "commercial_finish_fallback_final_source": _v63623_finish_fallback,
        "v60_applied_or_accepted": bool(_v63623_v60_applied),
        "rollback_to_standard_final": bool(_v63624_rollback_to_standard),
        "resolution_reason": _v63623_resolution_reason,
        "policy": "Debug brief and top-level report preserve authoritative report.final_source, but known stale standard labels are overridden when v60 is accepted/applied. If export/damage authority rolls back to the standard final, rollback-to-standard wins over any prior final_source label.",
    }
    # v63.6.2.11: current-version key plus backward-compatible aliases.
    report["final_source_authority_v6362_10"] = _v63624_final_source_authority
    report["final_source_authority_v6362_9"] = dict(_v63624_final_source_authority)
    report["final_source_authority_v6362_8"] = dict(_v63624_final_source_authority)
    report["final_source_authority_v6362_7"] = dict(_v63624_final_source_authority)
    report["final_source_authority_v6362_6"] = dict(_v63624_final_source_authority)
    report["final_source_authority_v6362_5"] = dict(_v63624_final_source_authority)
    report["final_source_authority_v6362_4"] = dict(_v63624_final_source_authority)
    # Backward-compatible aliases for dashboards/smoke tests added in v63.6.2.2/2.3.
    report["final_source_authority_v6362_3"] = dict(_v63624_final_source_authority)
    report["final_source_authority_v6362_2"] = dict(_v63624_final_source_authority)
    report["measurement_warning"] = auth.get("warning") or meas.get("warning")
    report["final_metric_aliases_v62_9_4_1"] = {
        "active": True,
        "source": "output_analysis_plus_commercial_loudness_finish",
        "true_peak_alias_authority": "approx_true_peak_dbfs mirrored to true_peak_dbtp/true_peak_dbfs",
        "policy": "Telemetry-only alias mirror; audio and selection are unchanged.",
    }
    report["true_peak_alias_consistency_v62_9_9"] = {
        "active": bool(tp is not None),
        "authority": "approx_true_peak_dbfs",
        "true_peak_dbtp": round(float(tp), 3) if tp is not None else None,
        "policy": "Final report/output_analysis true-peak aliases are normalized to one measured authority to avoid mismatched debug readings.",
    }
    return report



def _v6362_8_finalize_report_authority(report: dict[str, Any] | None, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """Final report-authority pass after legacy/framework report-only finalizers.

    v63.6.2.2~2.8 fixed individual final_source/target telemetry edge cases,
    but run_stage_limit/single_stage still attached the Research Capability
    Framework after the DML authority pass, allowing framework engine/schema
    labels to overwrite the delivered report-authority label.  v63.6.2.9 keeps
    this compatibility function name but stamps v63.6.2.9 and is called again
    after framework attachment.  It does not render audio, change limiter gain,
    or relax QC.
    """
    rep = report if isinstance(report, dict) else {}
    if not rep:
        return rep
    before_engine = rep.get("engine_version")
    before_schema = rep.get("mastering_report_schema_version")
    before_pipeline = rep.get("pipeline_version")
    finish = rep.get("commercial_loudness_finish") if isinstance(rep.get("commercial_loudness_finish"), dict) else {}
    final_analysis = rep.get("output_analysis") if isinstance(rep.get("output_analysis"), dict) else {}
    # v63.6.2.11: the final-order pass can run more than once: once before
    # Research Capability Framework attachment and once after it.  Re-running
    # the metric alias helper must not wash out the earlier, more informative
    # final_source_authority reason (for example stale_standard->v60 or
    # stale_v60->rollback_standard).  Capture existing authority before the
    # reapply pass and restore it when the delivered final_source is unchanged.
    before_authority = None
    for _key in ("final_source_authority_v6362_10", "final_source_authority_v6362_9", "final_source_authority_v6362_8", "final_source_authority_v6362_7"):
        if isinstance(rep.get(_key), dict):
            before_authority = dict(rep.get(_key) or {})
            break
    before_target_semantics = None
    for _key in ("target_semantics_v6362_10", "target_semantics_v6362_9", "target_semantics_v6362_8", "target_semantics_v6362_7"):
        if isinstance(rep.get(_key), dict):
            before_target_semantics = dict(rep.get(_key) or {})
            break
    authority_error = None
    try:
        rep = _v62_9_4_1_attach_top_level_final_metric_aliases(rep, final_analysis, finish)
    except Exception as exc:
        authority_error = f"{type(exc).__name__}: {str(exc)[:240]}"

    current_key = rep.get("final_source_authority_v6362_10") if isinstance(rep.get("final_source_authority_v6362_10"), dict) else None
    if current_key is None and isinstance(rep.get("final_source_authority_v6362_9"), dict):
        current_key = dict(rep.get("final_source_authority_v6362_9") or {})
        current_key["schema_version"] = "busy_v6362_10_final_source_authority"
        current_key["upconverted_from"] = "final_source_authority_v6362_9"
        rep["final_source_authority_v6362_10"] = current_key
    if current_key is None and isinstance(rep.get("final_source_authority_v6362_8"), dict):
        current_key = dict(rep.get("final_source_authority_v6362_8") or {})
        current_key["schema_version"] = "busy_v6362_10_final_source_authority"
        current_key["upconverted_from"] = "final_source_authority_v6362_8"
        rep["final_source_authority_v6362_10"] = current_key

    # Preserve a previously-resolved authority reason when this is merely a
    # final-order revalidation and the authoritative delivered source did not
    # change.  This keeps debug lineage stable instead of degrading to
    # `preserved_existing_top_level_report_final_source` on the second pass.
    if isinstance(before_authority, dict) and isinstance(current_key, dict):
        _before_src = before_authority.get("final_source")
        _current_src = current_key.get("final_source") or rep.get("final_source")
        _before_reason = before_authority.get("resolution_reason")
        _current_reason = current_key.get("resolution_reason")
        if _before_src and _before_src == _current_src and _before_reason and _before_reason != _current_reason:
            _preserved = dict(before_authority)
            _preserved["schema_version"] = "busy_v6362_10_final_source_authority"
            _preserved["revalidated_by_v6362_10_final_order_pass"] = True
            _preserved["current_reapply_resolution_reason"] = _current_reason
            _preserved["current_report_final_source_after_reapply"] = _current_src
            _idempotence_note = "v63.6.2.11 keeps previously resolved final_source authority idempotent across repeated report-authority finalizer passes."
            _existing_policy = str(_preserved.get("policy") or "").strip()
            _preserved["policy"] = (
                _existing_policy if _idempotence_note in _existing_policy
                else (_existing_policy + " " + _idempotence_note).strip()
            )
            rep["final_source_authority_v6362_10"] = _preserved
            for _alias in ("final_source_authority_v6362_9", "final_source_authority_v6362_8", "final_source_authority_v6362_7", "final_source_authority_v6362_6", "final_source_authority_v6362_5", "final_source_authority_v6362_4", "final_source_authority_v6362_3", "final_source_authority_v6362_2"):
                rep[_alias] = dict(_preserved)
            current_key = _preserved

    target_key = rep.get("target_semantics_v6362_10") if isinstance(rep.get("target_semantics_v6362_10"), dict) else None
    if target_key is None and isinstance(rep.get("target_semantics_v6362_9"), dict):
        target_key = dict(rep.get("target_semantics_v6362_9") or {})
        target_key["schema_version"] = "busy_v6362_10_target_source_telemetry"
        target_key["upconverted_from"] = "target_semantics_v6362_9"
        rep["target_semantics_v6362_10"] = target_key
    if target_key is None and isinstance(rep.get("target_semantics_v6362_8"), dict):
        target_key = dict(rep.get("target_semantics_v6362_8") or {})
        target_key["schema_version"] = "busy_v6362_10_target_source_telemetry"
        target_key["upconverted_from"] = "target_semantics_v6362_8"
        rep["target_semantics_v6362_10"] = target_key
    if isinstance(before_target_semantics, dict) and isinstance(target_key, dict):
        if before_target_semantics.get("final_lufs") == target_key.get("final_lufs") and before_target_semantics.get("target_lufs") == target_key.get("target_lufs"):
            target_key.setdefault("revalidated_by_v6362_10_final_order_pass", True)
            rep["target_semantics_v6362_10"] = target_key

    if before_engine and before_engine != V6362_MASTERING_ENGINE_VERSION:
        rep.setdefault("base_engine_version_before_v6362_10_report_authority", before_engine)
        rep.setdefault("base_engine_version_before_v6362_9_report_authority", before_engine)
    if before_schema and before_schema != "busy_master_report_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp":
        rep.setdefault("base_mastering_report_schema_version_before_v6362_10_report_authority", before_schema)
        rep.setdefault("base_mastering_report_schema_version_before_v6362_9_report_authority", before_schema)
    if before_pipeline and before_pipeline != "busy_auto_mastering_pipeline_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp":
        rep.setdefault("base_pipeline_version_before_v6362_10_report_authority", before_pipeline)
        rep.setdefault("base_pipeline_version_before_v6362_9_report_authority", before_pipeline)
    rep["engine_version"] = V6362_MASTERING_ENGINE_VERSION
    rep["pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
    rep["mastering_report_schema_version"] = "busy_master_report_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
    rep["v6362_10_report_authority_finalizer"] = {
        "schema_version": "busy_v6362_10_report_authority_finalizer",
        "active": True,
        "applied": True,
        "reason": "post_framework_final_order_idempotent_engine_schema_and_dml_authority_restamped",
        "before_engine_version": before_engine,
        "before_mastering_report_schema_version": before_schema,
        "before_pipeline_version": before_pipeline,
        "final_source": rep.get("final_source"),
        "final_source_authority_reason": (rep.get("final_source_authority_v6362_10") or {}).get("resolution_reason") if isinstance(rep.get("final_source_authority_v6362_10"), dict) else None,
        "authority_reason_revalidated": bool(isinstance(rep.get("final_source_authority_v6362_10"), dict) and rep["final_source_authority_v6362_10"].get("revalidated_by_v6362_10_final_order_pass")),
        "target_reached": (rep.get("target_semantics_v6362_10") or {}).get("target_reached") if isinstance(rep.get("target_semantics_v6362_10"), dict) else rep.get("target_reached"),
        "target_lufs_gap_lu": (rep.get("target_semantics_v6362_10") or {}).get("target_lufs_gap_lu") if isinstance(rep.get("target_semantics_v6362_10"), dict) else rep.get("target_lufs_gap_lu"),
        "authority_reapply_error": authority_error,
        "policy": "Report-only consolidation after legacy/framework finalizers. v63.7.0.1 keeps v6362_10 final_source/target authority payloads and restores explicit DML telemetry aliases while exposing the Body/Center/Vocal Support patch wrapper. No DSP, limiter gain, target threshold, warning threshold, or DTP intensity is changed.",
    }
    rep["v6362_9_report_authority_finalizer"] = dict(rep["v6362_10_report_authority_finalizer"])
    rep["v6362_8_report_authority_finalizer"] = dict(rep["v6362_10_report_authority_finalizer"])
    return rep



# ---------------------------------------------------------------------------
# v8.5.3.63.1.4.3: translation / floor-boundary crest QC guard.
# ---------------------------------------------------------------------------

def _v6314_3_num(v: Any, default: float | None = None) -> float | None:
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def _v6314_3_first_num(*vals: Any, default: float | None = None) -> float | None:
    for v in vals:
        x = _v6314_3_num(v, None)
        if x is not None:
            return x
    return default


def _v6314_3_ms_side_trim(y: np.ndarray, trim_db: float) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim == 1 or arr.shape[1] < 2:
        return np.asarray(y)
    out = arr.copy()
    side_gain = 10.0 ** (-max(0.0, float(trim_db)) / 20.0)
    mid = 0.5 * (out[:, 0] + out[:, 1])
    side = 0.5 * (out[:, 0] - out[:, 1]) * side_gain
    out[:, 0] = mid + side
    out[:, 1] = mid - side
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _v6314_3_analysis_delta(a: dict[str, Any] | None) -> float | None:
    a = a if isinstance(a, dict) else {}
    loud = a.get("loudness") if isinstance(a.get("loudness"), dict) else {}
    meas = a.get("measurement_authority") if isinstance(a.get("measurement_authority"), dict) else loud.get("measurement_authority") if isinstance(loud.get("measurement_authority"), dict) else {}
    direct = _v6314_3_first_num(a.get("stereo_minus_mono_lufs_db"), loud.get("stereo_minus_mono_lufs_db"), meas.get("stereo_minus_mono_lufs_db"))
    if direct is not None:
        return direct
    st = _v6314_3_first_num(a.get("stereo_integrated_lufs"), loud.get("stereo_integrated_lufs"), meas.get("stereo_integrated_lufs"), a.get("integrated_lufs"), loud.get("integrated_lufs"))
    mo = _v6314_3_first_num(a.get("mono_downmix_lufs"), loud.get("mono_downmix_lufs"), meas.get("mono_downmix_lufs"))
    if st is not None and mo is not None:
        return float(st) - float(mo)
    return None


def _v6314_3_apply_mono_translation_guard(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    finish_report: dict[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Runtime-safe final mono-translation guard.

    This is intentionally conservative: it only trims side energy when the
    delivered stereo master is much louder than its mono downmix, then accepts
    the candidate only if the commercial floor remains met and crest/LRA/TP do
    not regress.  It is a translation safety step, not a loudness chase.
    """
    enabled = _env_bool("BUSY_V6314_3_MONO_TRANSLATION_GUARD", True)
    rep: dict[str, Any] = {
        "schema_version": "busy_v6314_3_mono_translation_guard",
        "enabled": bool(enabled),
        "active": False,
        "applied": False,
        "policy": "If stereo-minus-mono LUFS gap is excessive, test a tiny M/S side trim and accept only when commercial floor, true peak, crest, and LRA remain safe.",
    }
    try:
        arr = np.asarray(y, dtype=np.float32)
        if not enabled:
            rep["reason"] = "disabled"
            return y, rep
        if arr.ndim == 1 or arr.shape[1] < 2:
            rep["reason"] = "mono_or_not_stereo"
            return y, rep
        hp = _v6314_handoff_policy_from_decision(decision)
        if not hp.get("active") and not _env_bool("BUSY_V6314_3_MONO_GUARD_WITHOUT_BAMIX", False):
            rep["reason"] = "not_bamix_handoff"
            return y, rep
        pre = _runtime_safe_final_analysis(arr, sr, None)
        delta = _v6314_3_analysis_delta(pre)
        floor = _v6314_3_first_num((finish_report or {}).get("commercial_floor_lufs"), hp.get("effective_floor_lufs"), default=None)
        pre_lufs = _v6314_3_first_num(pre.get("stereo_integrated_lufs"), pre.get("integrated_lufs"), default=None)
        pre_tp = _v6314_3_first_num(pre.get("approx_true_peak_dbfs"), pre.get("true_peak_dbfs"), default=None)
        pre_crest = _v6314_3_first_num(pre.get("crest_factor_db"), (pre.get("loudness") or {}).get("crest_factor_db") if isinstance(pre.get("loudness"), dict) else None, default=None)
        pre_lra = _analysis_lra_lu(pre)
        threshold = _env_float("BUSY_V6314_3_MONO_GAP_TRIGGER_DB", 2.50, 1.00, 6.00)
        rep.update({
            "active": bool(delta is not None and delta >= threshold),
            "before_stereo_minus_mono_lufs_db": round(float(delta), 3) if delta is not None else None,
            "trigger_db": round(float(threshold), 3),
            "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
            "before_lufs": round(float(pre_lufs), 3) if pre_lufs is not None else None,
            "before_true_peak_dbtp": round(float(pre_tp), 3) if pre_tp is not None else None,
            "before_crest_db": round(float(pre_crest), 3) if pre_crest is not None else None,
            "before_lra_lu": round(float(pre_lra), 3) if pre_lra is not None else None,
        })
        if delta is None:
            rep["reason"] = "missing_stereo_mono_delta"
            return y, rep
        if float(delta) < float(threshold):
            rep["reason"] = "stereo_mono_gap_below_trigger"
            return y, rep
        trim_db = min(
            _env_float("BUSY_V6314_3_MONO_SIDE_TRIM_MAX_DB", 0.95, 0.10, 1.75),
            _env_float("BUSY_V6314_3_MONO_SIDE_TRIM_BASE_DB", 0.38, 0.05, 1.00) + max(0.0, float(delta) - float(threshold)) * _env_float("BUSY_V6314_3_MONO_SIDE_TRIM_PER_DB", 0.34, 0.05, 0.90),
        )
        cand = _v6314_3_ms_side_trim(arr, trim_db)
        cand = lookahead_limiter(cand, sr, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=1.0, release_ms=120)
        cand = np.asarray(cand, dtype=np.float32)
        cand_an = _runtime_safe_final_analysis(cand, sr, None)
        cand_delta = _v6314_3_analysis_delta(cand_an)
        cand_lufs = _v6314_3_first_num(cand_an.get("stereo_integrated_lufs"), cand_an.get("integrated_lufs"), default=None)
        cand_tp = _v6314_3_first_num(cand_an.get("approx_true_peak_dbfs"), cand_an.get("true_peak_dbfs"), default=None)
        cand_crest = _v6314_3_first_num(cand_an.get("crest_factor_db"), (cand_an.get("loudness") or {}).get("crest_factor_db") if isinstance(cand_an.get("loudness"), dict) else None, default=None)
        cand_lra = _analysis_lra_lu(cand_an)
        makeup_db = 0.0
        min_margin = _env_float("BUSY_V6314_3_MONO_FLOOR_MARGIN_DB", 0.03, 0.00, 0.20)
        if floor is not None and cand_lufs is not None and cand_lufs < float(floor) + min_margin:
            need = float(floor) + min_margin - float(cand_lufs)
            tp_headroom = (float(FINAL_TRUE_PEAK_CEILING_DB) - float(cand_tp) - 0.05) if cand_tp is not None else 0.0
            makeup_db = max(0.0, min(need, tp_headroom, _env_float("BUSY_V6314_3_MONO_MAKEUP_MAX_DB", 0.35, 0.0, 0.8)))
            if makeup_db > 0.03:
                cand = (cand.astype(np.float64) * (10.0 ** (makeup_db / 20.0))).astype(np.float32)
                cand = lookahead_limiter(cand, sr, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=1.0, release_ms=120)
                cand_an = _runtime_safe_final_analysis(cand, sr, None)
                cand_delta = _v6314_3_analysis_delta(cand_an)
                cand_lufs = _v6314_3_first_num(cand_an.get("stereo_integrated_lufs"), cand_an.get("integrated_lufs"), default=None)
                cand_tp = _v6314_3_first_num(cand_an.get("approx_true_peak_dbfs"), cand_an.get("true_peak_dbfs"), default=None)
                cand_crest = _v6314_3_first_num(cand_an.get("crest_factor_db"), (cand_an.get("loudness") or {}).get("crest_factor_db") if isinstance(cand_an.get("loudness"), dict) else None, default=None)
                cand_lra = _analysis_lra_lu(cand_an)
        improvement = (float(delta) - float(cand_delta)) if cand_delta is not None else None
        floor_ok = True if floor is None or cand_lufs is None else bool(float(cand_lufs) >= float(floor) - _env_float("BUSY_V6314_3_MONO_FLOOR_TOL_DB", 0.02, 0.0, 0.10))
        tp_ok = True if cand_tp is None else bool(float(cand_tp) <= float(FINAL_TRUE_PEAK_CEILING_DB) + 0.03)
        crest_ok = True if pre_crest is None or cand_crest is None else bool(float(cand_crest) >= float(pre_crest) - _env_float("BUSY_V6314_3_MONO_CREST_MAX_REGRESS_DB", 0.18, 0.0, 0.75))
        lra_ok = True if pre_lra is None or cand_lra is None else bool(float(cand_lra) >= float(pre_lra) - _env_float("BUSY_V6314_3_MONO_LRA_MAX_REGRESS_LU", 0.12, 0.0, 0.50))
        improve_ok = bool(improvement is not None and improvement >= _env_float("BUSY_V6314_3_MONO_MIN_IMPROVE_DB", 0.18, 0.0, 1.00))
        accept = bool(improve_ok and floor_ok and tp_ok and crest_ok and lra_ok)
        rep.update({
            "side_trim_db": round(float(trim_db), 3),
            "makeup_db": round(float(makeup_db), 3),
            "after_stereo_minus_mono_lufs_db": round(float(cand_delta), 3) if cand_delta is not None else None,
            "improvement_db": round(float(improvement), 3) if improvement is not None else None,
            "after_lufs": round(float(cand_lufs), 3) if cand_lufs is not None else None,
            "after_true_peak_dbtp": round(float(cand_tp), 3) if cand_tp is not None else None,
            "after_crest_db": round(float(cand_crest), 3) if cand_crest is not None else None,
            "after_lra_lu": round(float(cand_lra), 3) if cand_lra is not None else None,
            "acceptance_checks": {
                "improve_ok": improve_ok,
                "floor_ok": floor_ok,
                "true_peak_ok": tp_ok,
                "crest_ok": crest_ok,
                "lra_ok": lra_ok,
            },
            "applied": accept,
            "reason": "accepted_mono_translation_side_trim" if accept else "candidate_rejected_guard_checks",
        })
        if accept:
            return cand, rep
        return y, rep
    except Exception as exc:
        rep.update({"active": False, "applied": False, "reason": "exception", "error_type": type(exc).__name__, "error": str(exc)[:300]})
        return y, rep


def _v6314_3_attach_handoff_qc_semantics(report: dict[str, Any] | None, decision: dict[str, Any] | None) -> dict[str, Any]:
    """Attach separated crest/target semantics after final metrics are known."""
    if not isinstance(report, dict):
        return report or {}
    try:
        hp = _v6314_handoff_policy_from_decision(decision)
        iam = decision.get("input_assist_modes") if isinstance(decision, dict) and isinstance(decision.get("input_assist_modes"), dict) else {}
        # v63.1.4.3.1: prefer report-level BAMix when it carries premaster_qc.
        # The decision mirror can be intentionally compact and lacks premaster_qc,
        # which caused the crest tradeoff semantic to miss available BAMix metrics.
        report_bamix = report.get("busy_auto_mixing") if isinstance(report.get("busy_auto_mixing"), dict) else {}
        decision_bamix = iam.get("busy_auto_mixing") if isinstance(iam.get("busy_auto_mixing"), dict) else {}
        bamix = report_bamix if isinstance(report_bamix.get("premaster_qc"), dict) else decision_bamix if isinstance(decision_bamix, dict) else report_bamix
        qc = bamix.get("premaster_qc") if isinstance(bamix.get("premaster_qc"), dict) else {}
        out = report.get("output_analysis") if isinstance(report.get("output_analysis"), dict) else {}
        fin = report.get("commercial_loudness_finish") if isinstance(report.get("commercial_loudness_finish"), dict) else {}
        proc = report.get("processing_report") if isinstance(report.get("processing_report"), dict) else {}
        lit = proc.get("loudness_iteration") if isinstance(proc.get("loudness_iteration"), dict) else {}
        tg = proc.get("v6314_2_bamix_loudness_target_guard") if isinstance(proc.get("v6314_2_bamix_loudness_target_guard"), dict) else lit.get("v6314_2_bamix_loudness_target_guard") if isinstance(lit.get("v6314_2_bamix_loudness_target_guard"), dict) else {}
        preferred = _v6314_3_first_num(fin.get("preferred_target_lufs"), hp.get("preferred_final_target_lufs"), tg.get("original_target_lufs"), fin.get("target_lufs"), default=None)
        actual = _v6314_3_first_num(report.get("actual_render_target_lufs"), tg.get("effective_target_lufs"), lit.get("target_lufs"), default=None)
        floor = _v6314_3_first_num(fin.get("commercial_floor_lufs"), hp.get("effective_floor_lufs"), tg.get("effective_floor_lufs"), report.get("commercial_floor_lufs"), default=None)
        final_lufs = _v6314_3_first_num(report.get("final_lufs"), out.get("integrated_lufs"), out.get("stereo_integrated_lufs"), default=None)
        final_crest = _v6314_3_first_num(report.get("crest_factor_db"), out.get("crest_factor_db"), (out.get("loudness") or {}).get("crest_factor_db") if isinstance(out.get("loudness"), dict) else None, default=None)
        premaster_crest = _v6314_3_first_num(qc.get("crest_factor_db"), default=None)
        premaster_lra = _v6314_3_first_num(qc.get("lra_lu"), default=None)
        final_lra = _v6314_3_first_num(report.get("lra_lu"), _analysis_lra_lu(out), default=None)
        premaster_loss = (float(premaster_crest) - float(final_crest)) if premaster_crest is not None and final_crest is not None else None
        premaster_lra_loss = (float(premaster_lra) - float(final_lra)) if premaster_lra is not None and final_lra is not None else None
        max_loss = _v6314_3_first_num(hp.get("max_crest_loss_db"), default=_env_float("BUSY_BAMIX_V6314_PUNCH_MAX_CREST_LOSS_DB", 2.85, 1.0, 4.0))
        floor_margin = (float(final_lufs) - float(floor)) if final_lufs is not None and floor is not None else None
        over_by = (float(premaster_loss) - float(max_loss)) if premaster_loss is not None and max_loss is not None else None
        boundary_tol = _env_float("BUSY_V6314_3_FLOOR_BOUNDARY_CREST_TRADEOFF_TOL_DB", 0.18, 0.0, 0.60)
        floor_boundary = bool(floor_margin is not None and -0.02 <= float(floor_margin) <= _env_float("BUSY_V6314_3_FLOOR_BOUNDARY_MARGIN_DB", 0.12, 0.0, 0.35))
        slight_over = bool(over_by is not None and 0.0 < float(over_by) <= boundary_tol)
        classification = "not_applicable"
        display_warning = None
        if hp.get("active") and premaster_loss is not None:
            if floor_boundary and slight_over:
                classification = "floor_boundary_crest_tradeoff_not_louder_chase"
                display_warning = "crest_loss_monitor_floor_boundary"
            elif max_loss is not None and float(premaster_loss) > float(max_loss):
                classification = "premaster_relative_crest_loss_over_policy"
                display_warning = "crest_loss_relative"
            else:
                classification = "premaster_relative_crest_within_policy"
        semantics = {
            "schema_version": "busy_v6314_3_floor_boundary_crest_semantics",
            "active": bool(hp.get("active")),
            "classification": classification,
            "preferred_target_lufs": round(float(preferred), 3) if preferred is not None else None,
            "actual_render_target_lufs": round(float(actual), 3) if actual is not None else None,
            "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
            "final_lufs": round(float(final_lufs), 3) if final_lufs is not None else None,
            "floor_margin_lu": round(float(floor_margin), 3) if floor_margin is not None else None,
            "premaster_crest_db": round(float(premaster_crest), 3) if premaster_crest is not None else None,
            "final_crest_db": round(float(final_crest), 3) if final_crest is not None else None,
            "premaster_to_final_crest_loss_db": round(float(premaster_loss), 3) if premaster_loss is not None else None,
            "max_premaster_crest_loss_db": round(float(max_loss), 3) if max_loss is not None else None,
            "crest_policy_over_by_db": round(float(over_by), 3) if over_by is not None else None,
            "handoff_mode": hp_mode or None,
            "density_completion_crest_grace_db": round(float(density_crest_grace), 3) if density_completion else None,
            "premaster_lra_lu": round(float(premaster_lra), 3) if premaster_lra is not None else None,
            "final_lra_lu": round(float(final_lra), 3) if final_lra is not None else None,
            "premaster_to_final_lra_loss_lu": round(float(premaster_lra_loss), 3) if premaster_lra_loss is not None else None,
            "display_warning": display_warning,
            "policy": "Separate original-input crest loss from BAMix-premaster-to-final crest loss; near-floor masters with tiny premaster cap overshoot are monitored as a floor-boundary tradeoff rather than treated as a louder-chase failure.",
        }
        report["v6314_3_floor_boundary_crest_semantics"] = semantics
        if preferred is not None:
            report["preferred_target_lufs"] = round(float(preferred), 3)
        if actual is not None:
            report["actual_render_target_lufs"] = round(float(actual), 3)
        fin["preferred_target_lufs"] = round(float(preferred), 3) if preferred is not None else fin.get("preferred_target_lufs")
        fin["actual_render_target_lufs"] = round(float(actual), 3) if actual is not None else fin.get("actual_render_target_lufs")
        fin["v6314_3_floor_boundary_crest_semantics"] = semantics
        # Keep raw adaptive QC intact, but provide a display-safe warning list for
        # debug/users when the only crest issue is the floor-boundary tradeoff.
        aq = report.get("adaptive_input_relative_qc_governance") if isinstance(report.get("adaptive_input_relative_qc_governance"), dict) else {}
        if display_warning == "crest_loss_monitor_floor_boundary" and isinstance(aq, dict):
            wr = aq.setdefault("warning_reclassification_v6314_3", {})
            raw = [str(x) for x in (report.get("remaining_safety_warnings") or [])]
            adj = [w for w in raw if w != "crest_loss_relative"]
            if display_warning not in adj:
                adj.append(display_warning)
            wr.update({
                "active": True,
                "reason": "floor_boundary_premaster_relative_crest_tradeoff",
                "raw_warnings": sorted(set(raw)),
                "display_warnings": sorted(set(adj)),
                "removed_warnings": ["crest_loss_relative"] if "crest_loss_relative" in raw else [],
                "added_warnings": [display_warning],
            })
            report["display_safety_warnings_v6314_3"] = sorted(set(adj))
        return report
    except Exception as exc:
        report["v6314_3_floor_boundary_crest_semantics"] = {"schema_version": "busy_v6314_3_floor_boundary_crest_semantics", "active": False, "reason": "exception", "error": str(exc)[:240]}
        return report



def _v6314_4_finalize_floor_tolerance_and_handoff_markers(report: dict[str, Any] | None, decision: dict[str, Any] | None = None) -> dict[str, Any]:
    """v63.1.4.4 report/QC finalizer for BAMix handoff edge cases.

    This is deliberately report-only.  It does not create a new render and it
    does not add gain.  It runs after late report attachments (especially
    Busy Auto Mixing) so premaster-relative crest semantics can be computed
    from the full report rather than the compact decision mirror.
    """
    if not isinstance(report, dict):
        return report or {}

    def _num(v: Any, default: float | None = None) -> float | None:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def _first(*vals: Any, default: float | None = None) -> float | None:
        for v in vals:
            x = _num(v, None)
            if x is not None:
                return x
        return default

    try:
        fin = report.get("commercial_loudness_finish") if isinstance(report.get("commercial_loudness_finish"), dict) else {}
        out = report.get("output_analysis") if isinstance(report.get("output_analysis"), dict) else {}
        loud = out.get("loudness") if isinstance(out.get("loudness"), dict) else {}
        proc = report.get("processing_report") if isinstance(report.get("processing_report"), dict) else {}
        lit = proc.get("loudness_iteration") if isinstance(proc.get("loudness_iteration"), dict) else {}
        tg = proc.get("v6314_2_bamix_loudness_target_guard") if isinstance(proc.get("v6314_2_bamix_loudness_target_guard"), dict) else lit.get("v6314_2_bamix_loudness_target_guard") if isinstance(lit.get("v6314_2_bamix_loudness_target_guard"), dict) else {}
        bamix = report.get("busy_auto_mixing") if isinstance(report.get("busy_auto_mixing"), dict) else {}
        qc = bamix.get("premaster_qc") if isinstance(bamix.get("premaster_qc"), dict) else {}
        hp = _v6314_handoff_policy_from_decision(decision if isinstance(decision, dict) else {})
        # v63.1.9.4: this finalizer can run after BAMix metadata has been
        # attached to the report, while the compact decision object available in
        # this call may still be an earlier mirror.  Prefer report-attached
        # handoff policy for final QC semantics so density-completion does not
        # fall back to old transparent-polish/over-policy labels.
        try:
            _ia_for_hp = report.get("input_analysis") if isinstance(report.get("input_analysis"), dict) else {}
            _bamix_for_hp = report.get("busy_auto_mixing") if isinstance(report.get("busy_auto_mixing"), dict) else {}
            _ia_bamix_for_hp = _ia_for_hp.get("busy_auto_mixing") if isinstance(_ia_for_hp.get("busy_auto_mixing"), dict) else {}
            _hp_candidates = [
                report.get("busy_auto_mixing_mastering_handoff_policy"),
                _ia_for_hp.get("busy_auto_mixing_mastering_handoff_policy") if isinstance(_ia_for_hp, dict) else None,
                _bamix_for_hp.get("mastering_handoff_policy") if isinstance(_bamix_for_hp, dict) else None,
                _ia_bamix_for_hp.get("mastering_handoff_policy") if isinstance(_ia_bamix_for_hp, dict) else None,
            ]
            for _rhp in _hp_candidates:
                if isinstance(_rhp, dict) and _rhp.get("active"):
                    _mode_current = str(hp.get("handoff_mode") or hp.get("profile") or "").lower() if isinstance(hp, dict) else ""
                    _mode_report = str(_rhp.get("handoff_mode") or _rhp.get("profile") or "").lower()
                    if (not isinstance(hp, dict)) or (not hp.get("active")) or (not _mode_current) or ("density" in _mode_report and "density" not in _mode_current):
                        _merged_hp = copy.deepcopy(_rhp)
                        if isinstance(hp, dict):
                            for _k, _v in hp.items():
                                _merged_hp.setdefault(_k, _v)
                        hp = _merged_hp
                    else:
                        for _k, _v in _rhp.items():
                            hp.setdefault(_k, _v)
                    break
        except Exception:
            pass
        bamix_active = bool(
            hp.get("active")
            or tg.get("active")
            or bamix.get("premaster_used_for_mastering")
            or str(bamix.get("mastering_input_source") or "") == "busy_auto_mixing_premaster"
        )

        final_lufs = _first(report.get("final_lufs"), out.get("stereo_integrated_lufs"), loud.get("stereo_integrated_lufs"), out.get("integrated_lufs"), loud.get("integrated_lufs"))
        # v63.1.9.3: if the delivered final is the standard limiter output but
        # commercial_loudness_finish still points at a quieter evaluated
        # candidate, normalize the report so readers do not interpret the safe
        # checkpoint as the delivered master.
        try:
            _cand_lufs = _first((fin.get("output") or {}).get("integrated_lufs") if isinstance(fin.get("output"), dict) else None, (fin.get("output") or {}).get("stereo_integrated_lufs") if isinstance(fin.get("output"), dict) else None)
            _selected_label = str(fin.get("selected") or "")
            _standard_preserve_already_decided = bool(fin.get("candidate_rejected_against_standard_final")) or "standard_final_preserved_candidate_not_louder" in str(fin.get("reason") or "") or "candidate_not_louder_than_standard_final" in str(fin.get("candidate_rejection_reason") or "")
            _candidate_quieter_than_standard = bool(_cand_lufs is not None and final_lufs is not None and float(_cand_lufs) < float(final_lufs) - 0.12)
            if _selected_label and _selected_label != "standard_final_limiter_output" and (_candidate_quieter_than_standard or _standard_preserve_already_decided):
                fin.setdefault("rejected_candidate_label", _selected_label)
                if _candidate_quieter_than_standard and not isinstance(fin.get("rejected_candidate_output"), dict):
                    fin["rejected_candidate_output"] = copy.deepcopy(fin.get("output")) if isinstance(fin.get("output"), dict) else {}
                if _cand_lufs is not None and final_lufs is not None:
                    fin["candidate_vs_standard_lufs_delta_db"] = round(float(_cand_lufs) - float(final_lufs), 3)
                fin["candidate_rejected_against_standard_final"] = True
                fin["candidate_rejection_reason"] = "candidate_not_louder_than_standard_final_standard_preserved_target_met" if bool(report.get("target_met") or fin.get("target_met")) else "candidate_not_louder_than_standard_final"
                fin["selected_before_v6319_4_standard_preserve"] = _selected_label
                fin["selected"] = "standard_final_limiter_output"
                fin["selected_final_source"] = "standard_final_limiter_output"
                fin["fallback_final_source"] = "standard_final_limiter_output"
                fin["selected_candidate_replaced_standard_final_limiter"] = False
                fin["delivered_final_is_standard_after_v6319_4_report_normalizer"] = True
                fin["reason_before_v6319_4_standard_preserve"] = fin.get("reason")
                fin["reason"] = "standard_final_preserved_candidate_not_louder_target_met" if bool(report.get("target_met") or fin.get("target_met")) else "standard_final_preserved_candidate_not_louder"
                fin["output"] = _analysis_summary_for_loudness(out) if isinstance(out, dict) else {}
                fin["delivered_final_output"] = _analysis_summary_for_loudness(out) if isinstance(out, dict) else {}
                report["commercial_loudness_finish"] = fin
        except Exception:
            pass

        floor = _first(fin.get("commercial_floor_lufs"), report.get("commercial_floor_lufs"), tg.get("effective_floor_lufs"), hp.get("effective_floor_lufs"))
        preferred = _first(report.get("preferred_target_lufs"), fin.get("preferred_target_lufs"), tg.get("original_target_lufs"), hp.get("preferred_final_target_lufs"), fin.get("target_lufs"))
        actual = _first(report.get("actual_render_target_lufs"), fin.get("actual_render_target_lufs"), tg.get("effective_target_lufs"), lit.get("target_lufs"))
        tp = _first(report.get("true_peak_dbtp"), out.get("approx_true_peak_dbfs"), loud.get("approx_true_peak_dbfs"), out.get("true_peak_dbfs"), loud.get("true_peak_dbfs"))
        ceiling = _first(report.get("final_true_peak_target_dbtp"), fin.get("reference_true_peak_ceiling_db"), tg.get("ceiling_db"), default=float(FINAL_TRUE_PEAK_CEILING_DB))
        final_crest = _first(report.get("crest_factor_db"), out.get("crest_factor_db"), loud.get("crest_factor_db"))
        final_lra = _first(report.get("lra_lu"), out.get("lra_lu"), _analysis_lra_lu(out))
        premaster_crest = _first(qc.get("crest_factor_db"), qc.get("crest_db"))
        premaster_lra = _first(qc.get("lra_lu"))
        max_loss = _first(hp.get("max_crest_loss_db"), default=_env_float("BUSY_BAMIX_V6314_PUNCH_MAX_CREST_LOSS_DB", 2.85, 1.0, 4.0))
        premaster_loss = (float(premaster_crest) - float(final_crest)) if premaster_crest is not None and final_crest is not None else None
        premaster_lra_loss = (float(premaster_lra) - float(final_lra)) if premaster_lra is not None and final_lra is not None else None
        floor_margin = (float(final_lufs) - float(floor)) if final_lufs is not None and floor is not None else None
        over_by = (float(premaster_loss) - float(max_loss)) if premaster_loss is not None and max_loss is not None else None

        # Surface target aliases after BAMix has been attached by master_job.
        if preferred is not None:
            report["preferred_target_lufs"] = round(float(preferred), 3)
            fin["preferred_target_lufs"] = round(float(preferred), 3)
        if actual is not None:
            report["actual_render_target_lufs"] = round(float(actual), 3)
            fin["actual_render_target_lufs"] = round(float(actual), 3)

        # Treat tiny near-floor misses as met-with-tolerance when a safer extra
        # push is not available.  Do not add gain; this only changes the boolean
        # authority and records the exact measured miss.
        floor_tol = _env_float("BUSY_V6314_4_COMMERCIAL_FLOOR_TOLERANCE_LU", 0.03, 0.0, 0.10)
        tp_no_room_db = _env_float("BUSY_V6314_4_TRUE_PEAK_NO_SAFE_ROOM_DB", 0.06, 0.0, 0.30)
        tp_headroom = (float(ceiling) - float(tp)) if tp is not None and ceiling is not None else None
        exact_floor_met = bool(final_lufs is not None and floor is not None and float(final_lufs) >= float(floor))
        within_floor_tol = bool(final_lufs is not None and floor is not None and float(final_lufs) >= float(floor) - float(floor_tol))
        no_safe_push_room = bool((tp_headroom is not None and float(tp_headroom) <= float(tp_no_room_db)) or tg.get("applied") or bamix_active)
        tol_accept = bool((not exact_floor_met) and within_floor_tol and no_safe_push_room)
        floor_tol_report = {
            "schema_version": "busy_v6314_4_commercial_floor_tolerance",
            "active": bool(final_lufs is not None and floor is not None),
            "applied": bool(tol_accept),
            "target_met_exact": bool(exact_floor_met) if floor is not None and final_lufs is not None else None,
            "target_met_with_tolerance": bool(exact_floor_met or tol_accept) if floor is not None and final_lufs is not None else None,
            "final_lufs": round(float(final_lufs), 3) if final_lufs is not None else None,
            "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
            "floor_margin_lu": round(float(floor_margin), 3) if floor_margin is not None else None,
            "tolerance_lu": round(float(floor_tol), 3),
            "true_peak_dbtp": round(float(tp), 3) if tp is not None else None,
            "true_peak_ceiling_dbtp": round(float(ceiling), 3) if ceiling is not None else None,
            "true_peak_headroom_db": round(float(tp_headroom), 3) if tp_headroom is not None else None,
            "no_safe_push_room": bool(no_safe_push_room),
            "reason": "floor_met_within_tolerance_no_extra_push" if tol_accept else ("floor_exactly_met" if exact_floor_met else "outside_floor_tolerance"),
            "policy": "Report-only floor tolerance for tiny LUFS misses near true-peak/BAMix handoff boundary; never adds gain or rerenders.",
        }
        report["v6314_4_commercial_floor_tolerance"] = floor_tol_report
        fin["v6314_4_commercial_floor_tolerance"] = floor_tol_report
        if tol_accept:
            fin["target_met_before_v6314_4_floor_tolerance"] = fin.get("target_met")
            fin["target_met_exact"] = False
            fin["target_met"] = True
            fin["reason_before_v6314_4_floor_tolerance"] = fin.get("reason")
            fin["reason"] = "commercial_floor_met_within_v6314_4_tolerance_no_safe_extra_push"
            report["target_met"] = True
            clr = report.get("commercial_loudness_result") if isinstance(report.get("commercial_loudness_result"), dict) else {}
            if isinstance(clr, dict):
                clr["target_met_before_v6314_4_floor_tolerance"] = clr.get("target_met")
                clr["target_met"] = True
                clr["reason_before_v6314_4_floor_tolerance"] = clr.get("reason")
                clr["reason"] = fin["reason"]
                clr["v6314_4_commercial_floor_tolerance"] = floor_tol_report
                report["commercial_loudness_result"] = clr

        boundary_tol = _env_float("BUSY_V6314_4_FLOOR_BOUNDARY_CREST_TRADEOFF_TOL_DB", 0.20, 0.0, 0.80)
        boundary_margin = _env_float("BUSY_V6314_4_FLOOR_BOUNDARY_MARGIN_LU", 0.12, 0.0, 0.35)
        lower_floor_margin = -float(floor_tol)
        floor_boundary = bool(floor_margin is not None and lower_floor_margin <= float(floor_margin) <= float(boundary_margin))
        slight_over = bool(over_by is not None and 0.0 < float(over_by) <= float(boundary_tol))
        hp_mode = str(hp.get("handoff_mode") or hp.get("profile") or "").lower()
        density_completion = bool(hp.get("active") and "density" in hp_mode)
        density_crest_grace = _env_float("BUSY_BAMIX_V6319_ADAPTIVE_QC_CREST_GRACE_DB", 0.18, 0.0, 0.60)
        density_near_over = bool(density_completion and over_by is not None and 0.0 < float(over_by) <= float(density_crest_grace))
        classification = "not_applicable"
        display_warning = None
        if bamix_active and premaster_loss is not None:
            if density_near_over:
                classification = "bamix_density_completion_crest_near_policy_boundary"
                display_warning = "bamix_crest_boundary_soft_warning"
            elif floor_boundary and slight_over:
                classification = "floor_boundary_crest_tradeoff_not_louder_chase"
                display_warning = "crest_loss_monitor_floor_boundary"
            elif max_loss is not None and float(premaster_loss) > float(max_loss):
                classification = "premaster_relative_crest_loss_over_policy"
                display_warning = "crest_loss_relative"
            else:
                classification = "premaster_relative_crest_within_policy"
        semantics = {
            "schema_version": "busy_v6314_4_floor_boundary_crest_semantics",
            "active": bool(bamix_active),
            "classification": classification,
            "preferred_target_lufs": round(float(preferred), 3) if preferred is not None else None,
            "actual_render_target_lufs": round(float(actual), 3) if actual is not None else None,
            "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
            "final_lufs": round(float(final_lufs), 3) if final_lufs is not None else None,
            "floor_margin_lu": round(float(floor_margin), 3) if floor_margin is not None else None,
            "floor_boundary": bool(floor_boundary),
            "premaster_crest_db": round(float(premaster_crest), 3) if premaster_crest is not None else None,
            "final_crest_db": round(float(final_crest), 3) if final_crest is not None else None,
            "premaster_to_final_crest_loss_db": round(float(premaster_loss), 3) if premaster_loss is not None else None,
            "max_premaster_crest_loss_db": round(float(max_loss), 3) if max_loss is not None else None,
            "crest_policy_over_by_db": round(float(over_by), 3) if over_by is not None else None,
            "premaster_lra_lu": round(float(premaster_lra), 3) if premaster_lra is not None else None,
            "final_lra_lu": round(float(final_lra), 3) if final_lra is not None else None,
            "premaster_to_final_lra_loss_lu": round(float(premaster_lra_loss), 3) if premaster_lra_loss is not None else None,
            "display_warning": display_warning,
            "policy": "Final attach runs after BAMix report handoff; original-input crest loss remains measurable, while BAMix premaster-to-final crest is the handoff quality authority for punch-preserved floor-boundary renders.",
        }
        report["v6314_4_floor_boundary_crest_semantics"] = semantics
        report["v6314_3_floor_boundary_crest_semantics"] = semantics  # compatibility alias
        fin["v6314_4_floor_boundary_crest_semantics"] = semantics
        fin["v6314_3_floor_boundary_crest_semantics"] = semantics

        raw_warnings = [str(x) for x in (report.get("remaining_safety_warnings") or [])]
        display_warnings = list(raw_warnings)
        if display_warning in {"crest_loss_monitor_floor_boundary", "bamix_crest_boundary_soft_warning"}:
            display_warnings = [w for w in display_warnings if w != "crest_loss_relative"]
            if display_warning not in display_warnings:
                display_warnings.append(display_warning)
            if display_warning == "bamix_crest_boundary_soft_warning" and "crest_loss_monitor" not in display_warnings:
                display_warnings.append("crest_loss_monitor")
            report["remaining_safety_warnings_before_v6314_4_crest_semantics"] = sorted(set(raw_warnings))
            report["remaining_safety_warnings"] = sorted(set(display_warnings))
            aq = report.get("adaptive_input_relative_qc_governance") if isinstance(report.get("adaptive_input_relative_qc_governance"), dict) else {}
            if isinstance(aq, dict):
                aq["warning_reclassification_v6314_4"] = {
                    "active": True,
                    "reason": "bamix_density_completion_crest_boundary" if display_warning == "bamix_crest_boundary_soft_warning" else "floor_boundary_premaster_relative_crest_tradeoff",
                    "raw_warnings": sorted(set(raw_warnings)),
                    "display_warnings": sorted(set(display_warnings)),
                    "removed_warnings": ["crest_loss_relative"] if "crest_loss_relative" in raw_warnings else [],
                    "added_warnings": [display_warning],
                    "authority": "v6314_4_floor_boundary_crest_semantics",
                }
                report["adaptive_input_relative_qc_governance"] = aq
        report["display_safety_warnings_v6314_4"] = sorted(set(display_warnings))

        # v63.1.9.4: Re-run the report-only adaptive QC after BAMix handoff
        # metadata has been attached.  Earlier passes may have run before the
        # readiness-aware policy was visible and left stale collapse/over-policy
        # labels in the final report.  This is telemetry-only and does not
        # render or change audio.
        try:
            if bamix_active and density_completion and isinstance(out, dict) and out:
                _decision_for_qc = copy.deepcopy(decision) if isinstance(decision, dict) else {}
                if isinstance(hp, dict) and hp.get("active"):
                    _decision_for_qc["busy_auto_mixing_mastering_handoff_policy"] = copy.deepcopy(hp)
                    _ai_for_qc = _decision_for_qc.setdefault("auto_intensity_policy", {})
                    if isinstance(_ai_for_qc, dict):
                        _ai_for_qc["bamix_premaster_handoff_policy"] = copy.deepcopy(hp)
                _stale_lra_labels = {"mastering_induced_lra_collapse", "lra_loss_relative"}
                _current_warnings_for_qc = [str(w) for w in (report.get("remaining_safety_warnings") or []) if str(w) not in _stale_lra_labels]
                _before_for_qc = report.get("input_analysis") if isinstance(report.get("input_analysis"), dict) else report.get("original_input_analysis_before_pre_clean")
                _mode_for_qc = str(report.get("mode") or _decision_for_qc.get("processing_mode") or _decision_for_qc.get("mastering_intensity") or "")
                _aq_reconciled = _adaptive_input_relative_qc_governance(
                    _before_for_qc,
                    out,
                    _decision_for_qc,
                    _mode_for_qc,
                    current_reasons=_current_warnings_for_qc,
                )
                if isinstance(_aq_reconciled, dict) and _aq_reconciled.get("active"):
                    _aq_reconciled["late_reconciled_v6319_4"] = {
                        "active": True,
                        "reason": "bamix_density_completion_policy_visible_after_report_attachment",
                        "authority": "v6314_4_handoff_qc_finalizer",
                    }
                    report["adaptive_input_relative_qc_governance"] = _aq_reconciled
                    if isinstance(_aq_reconciled.get("input_relative_damage_scoring"), dict):
                        report["adaptive_input_relative_damage_scoring"] = _aq_reconciled.get("input_relative_damage_scoring")
                    _updated = [str(w) for w in (_aq_reconciled.get("updated_warnings") or [])]
                    if _updated:
                        report["remaining_safety_warnings_before_v6319_4_adaptive_reconcile"] = sorted(set(report.get("remaining_safety_warnings") or []))
                        report["remaining_safety_warnings"] = sorted(set(_updated))
                        report["display_safety_warnings_v6314_4"] = sorted(set(_updated))
                        # Keep common aliases from reintroducing the old hard
                        # collapse labels in debug/report summaries.
                        for _warn_key in ("final_warnings", "safety_warnings"):
                            if isinstance(report.get(_warn_key), list):
                                report[_warn_key] = sorted(set([str(w) for w in report.get(_warn_key, []) if str(w) not in _stale_lra_labels] + _updated))
        except Exception as _v6319_4_exc:
            report["v6319_4_adaptive_qc_reconcile_error"] = str(_v6319_4_exc)[:240]

        # Always surface mono-translation status.  If the DSP guard did not
        # leave a marker (older deployment path or rejected before attach), add
        # a visibility marker so debug/Supabase can see why no side trim changed
        # the delivered buffer.
        mtg = report.get("v6314_3_mono_translation_guard") if isinstance(report.get("v6314_3_mono_translation_guard"), dict) else fin.get("v6314_3_mono_translation_guard") if isinstance(fin.get("v6314_3_mono_translation_guard"), dict) else {}
        delta = _first(report.get("stereo_minus_mono_lufs_db"), (fin.get("final_measurement_authority") or {}).get("stereo_minus_mono_lufs_db") if isinstance(fin.get("final_measurement_authority"), dict) else None)
        mono_threshold = _env_float("BUSY_V6314_3_MONO_GAP_TRIGGER_DB", 2.50, 1.00, 6.00)
        if not mtg:
            active = bool(delta is not None and float(delta) >= float(mono_threshold))
            if active and (tol_accept or (floor_margin is not None and float(floor_margin) <= float(floor_tol)) or (tp_headroom is not None and float(tp_headroom) <= float(tp_no_room_db))):
                reason = "blocked_by_floor_tp_no_safe_room_visibility_backfill"
            elif active:
                reason = "guard_candidate_not_recorded_visibility_backfill"
            elif delta is None:
                reason = "missing_stereo_mono_delta_visibility_backfill"
            else:
                reason = "stereo_mono_gap_below_trigger"
            mtg = {
                "schema_version": "busy_v6314_4_mono_translation_guard_visibility",
                "enabled": bool(_env_bool("BUSY_V6314_3_MONO_TRANSLATION_GUARD", True)),
                "active": bool(active),
                "applied": False,
                "visibility_backfill": True,
                "before_stereo_minus_mono_lufs_db": round(float(delta), 3) if delta is not None else None,
                "trigger_db": round(float(mono_threshold), 3),
                "final_lufs": round(float(final_lufs), 3) if final_lufs is not None else None,
                "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
                "floor_margin_lu": round(float(floor_margin), 3) if floor_margin is not None else None,
                "true_peak_dbtp": round(float(tp), 3) if tp is not None else None,
                "true_peak_ceiling_dbtp": round(float(ceiling), 3) if ceiling is not None else None,
                "true_peak_headroom_db": round(float(tp_headroom), 3) if tp_headroom is not None else None,
                "reason": reason,
                "policy": "Visibility-only marker.  If no safe floor/TP headroom exists, mono translation repair is reported as blocked rather than silently missing.",
            }
        else:
            mtg = copy.deepcopy(mtg)
            mtg["v6314_4_visibility_audit"] = {
                "active": True,
                "reason": mtg.get("reason"),
                "final_lufs": round(float(final_lufs), 3) if final_lufs is not None else None,
                "commercial_floor_lufs": round(float(floor), 3) if floor is not None else None,
                "floor_margin_lu": round(float(floor_margin), 3) if floor_margin is not None else None,
                "true_peak_headroom_db": round(float(tp_headroom), 3) if tp_headroom is not None else None,
            }
        report["v6314_4_mono_translation_guard"] = mtg
        report["v6314_3_mono_translation_guard"] = mtg  # compatibility alias
        fin["v6314_4_mono_translation_guard"] = mtg
        fin["v6314_3_mono_translation_guard"] = mtg
        report["commercial_loudness_finish"] = fin

        report["v6314_4_handoff_qc_finalizer"] = {
            "schema_version": "busy_v6314_4_1_handoff_qc_finalizer",
            "active": True,
            "applied": True,
            "reason": "floor_tolerance_crest_semantics_mono_visibility_finalized",
            "floor_tolerance_attached": isinstance(report.get("v6314_4_commercial_floor_tolerance"), dict),
            "crest_semantics_attached": isinstance(report.get("v6314_4_floor_boundary_crest_semantics"), dict),
            "mono_translation_guard_attached": isinstance(report.get("v6314_4_mono_translation_guard"), dict),
            "policy": "Report-only finalizer success marker; no audio rerender or gain push.",
        }
        report["engine_version"] = V6362_MASTERING_ENGINE_VERSION
        report["pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_63_3_4_1_aug_telemetry_reason_cleanup"
        report["mastering_report_schema_version"] = "busy_master_report_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
        return report
    except Exception as exc:
        report["v6314_4_handoff_qc_finalizer"] = {
            "schema_version": "busy_v6314_4_handoff_qc_finalizer",
            "active": False,
            "reason": "exception",
            "error_type": type(exc).__name__,
            "error": str(exc)[:400],
        }
        report["engine_version"] = V6362_MASTERING_ENGINE_VERSION
        report["pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_63_3_4_1_aug_telemetry_reason_cleanup"
        report["mastering_report_schema_version"] = "busy_master_report_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
        return report


# ---------------------------------------------------------------------------
# v8.5.3.57: Commercial Loudness Feasibility & Damage-Aware Safe Selection.
# This layer does not chase target LUFS directly.  It estimates whether the
# current stereo master has any safe push budget left, and rejects/penalizes
# candidates that gain loudness by collapsing LRA or damaging transients.
# ---------------------------------------------------------------------------

def _v57_analysis_metric(analysis: dict[str, Any] | None, *keys: str, default: float | None = None) -> float | None:
    if not isinstance(analysis, dict):
        return default
    for key in keys:
        try:
            value = analysis.get(key)
            if value is not None and np.isfinite(float(value)):
                return float(value)
        except Exception:
            pass
    if "lra" in " ".join(keys).lower():
        val = _analysis_lra_lu(analysis)
        if val is not None:
            return float(val)
    return default


def _v57_safe_push_feasibility(
    before: dict[str, Any] | None,
    current: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    floor_lufs: float,
    target_lufs: float,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Estimate safe remaining loudness budget before rendering more candidates.

    The target/floor are references, not commands.  The returned budget is a
    maximum candidate push budget allowed by LRA, transient/crest and true-peak
    conditions.  It intentionally becomes zero when the source is already too
    flat or transient-damaged, so the engine can choose the best safe candidate
    instead of forcing a louder but damaged master.
    """
    enabled = _env_bool("BUSY_V57_DAMAGE_AWARE_SAFE_SELECTION", True)
    before = before if isinstance(before, dict) else {}
    current = current if isinstance(current, dict) else {}
    profile = profile if isinstance(profile, dict) else {}
    warnings = sorted(set(warnings or []))
    cur_lufs = _v57_analysis_metric(current, "integrated_lufs", "lufs", default=-99.0)
    cur_tp = _v57_analysis_metric(current, "approx_true_peak_dbfs", "true_peak_dbtp", default=-99.0)
    cur_crest = _v57_analysis_metric(current, "crest_factor_db", "crest_db", default=99.0)
    cur_lra = _analysis_lra_lu(current)
    in_lra = _analysis_lra_lu(before)
    in_crest = _v57_analysis_metric(before, "crest_factor_db", "crest_db", default=cur_crest)
    lra_delta = None if (in_lra is None or cur_lra is None) else float(cur_lra) - float(in_lra)
    crest_delta = None if (in_crest is None or cur_crest is None) else float(cur_crest) - float(in_crest)
    hard_lra_floor = _env_float("BUSY_V57_HARD_LRA_FLOOR_LU", 1.05, 0.2, 4.0)
    soft_lra_floor = _env_float("BUSY_V57_SOFT_LRA_FLOOR_LU", 1.55, 0.4, 5.0)
    max_lra_loss = _env_float("BUSY_V57_MAX_LRA_LOSS_LU", 2.25, 0.4, 8.0)
    max_crest_loss = _env_float("BUSY_V57_MAX_CREST_LOSS_DB", 4.0, 0.8, 10.0)
    hard_crest_floor = float(profile.get("hard_crest_min_db", 4.5) or 4.5)
    tp_ceiling = float(COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB)
    reasons: list[str] = []
    budgets: dict[str, float] = {}

    gap_to_floor = max(0.0, float(floor_lufs) - float(cur_lufs or -99.0))
    gap_to_target = max(0.0, float(target_lufs) - float(cur_lufs or -99.0))
    # True peak budget is deliberately conservative because the limiter/clipper
    # may not translate input gain to LUFS 1:1, but TP overrun is never allowed.
    tp_margin = float(tp_ceiling) - float(cur_tp or -99.0)
    budgets["tp_db"] = max(0.0, min(1.6, tp_margin + 0.35)) if np.isfinite(tp_margin) else 0.0
    if cur_lra is None:
        budgets["lra_db"] = 0.45
        reasons.append("lra_unknown_conservative_budget")
    elif float(cur_lra) < hard_lra_floor:
        budgets["lra_db"] = 0.0
        reasons.append("lra_already_below_hard_floor")
    elif float(cur_lra) < soft_lra_floor:
        budgets["lra_db"] = 0.25
        reasons.append("lra_near_flat_floor")
    else:
        budgets["lra_db"] = 1.1
    if lra_delta is not None and float(lra_delta) < -max_lra_loss:
        budgets["lra_db"] = min(budgets.get("lra_db", 0.0), 0.0)
        reasons.append("lra_loss_budget_exhausted")
    if cur_crest is None:
        budgets["transient_db"] = 0.45
        reasons.append("crest_unknown_conservative_budget")
    elif float(cur_crest) < hard_crest_floor + 0.15:
        budgets["transient_db"] = 0.0
        reasons.append("crest_already_near_hard_floor")
    elif "transient_loss_delta" in warnings:
        budgets["transient_db"] = 0.0
        reasons.append("transient_loss_warning_budget_exhausted")
    elif crest_delta is not None and float(crest_delta) < -max_crest_loss:
        budgets["transient_db"] = 0.0
        reasons.append("crest_loss_budget_exhausted")
    else:
        budgets["transient_db"] = 1.05
    if "lra_too_flat" in warnings or "lra_reduced_too_much" in warnings:
        budgets["lra_db"] = min(budgets.get("lra_db", 0.0), 0.15)
        reasons.append("lra_warning_budget_limited")
    budget = max(0.0, min(float(v) for v in budgets.values())) if budgets else 0.0
    useful = bool(enabled and budget >= _env_float("BUSY_V57_MIN_USEFUL_SAFE_PUSH_DB", 0.18, 0.0, 0.8))
    return {
        "schema_version": "busy_v57_commercial_loudness_feasibility_v8_5_3_57_2",
        "enabled": bool(enabled),
        "active": bool(enabled),
        "policy": "target/floor are feasibility references; push budget is capped by LRA, transient/crest and true-peak safety, not forced LUFS chasing.",
        "current_lufs": round(float(cur_lufs), 3) if cur_lufs is not None else None,
        "target_lufs": round(float(target_lufs), 3),
        "commercial_floor_lufs": round(float(floor_lufs), 3),
        "gap_to_floor_lu": round(float(gap_to_floor), 3),
        "gap_to_target_lu": round(float(gap_to_target), 3),
        "current_true_peak_dbtp": round(float(cur_tp), 3) if cur_tp is not None else None,
        "current_lra_lu": round(float(cur_lra), 3) if cur_lra is not None else None,
        "input_lra_lu": round(float(in_lra), 3) if in_lra is not None else None,
        "lra_delta_lu": round(float(lra_delta), 3) if lra_delta is not None else None,
        "current_crest_db": round(float(cur_crest), 3) if cur_crest is not None else None,
        "input_crest_db": round(float(in_crest), 3) if in_crest is not None else None,
        "crest_delta_db": round(float(crest_delta), 3) if crest_delta is not None else None,
        "budget_by_axis_db": {k: round(float(v), 3) for k, v in budgets.items()},
        "safe_push_budget_db": round(float(budget), 3),
        "useful_safe_push_available": useful,
        "reasons": sorted(set(reasons)) or ["safe_push_budget_available"],
        "decision": "allow_bounded_safe_push" if useful else "do_not_push_select_best_safe_candidate",
    }


def _v57_damage_aware_candidate_gate(
    before: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
    base_analysis: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    warnings: list[str] | None,
    floor_lufs: float,
    target_lufs: float,
) -> dict[str, Any]:
    """Hard-gate candidates that gain loudness through audible damage.

    v56 ensured the selected buffer is the buffer being exported.  v57 ensures
    that the selected buffer is not accepted merely because it is louder when it
    is also measurably flat/transient-damaged.
    """
    enabled = _env_bool("BUSY_V57_DAMAGE_AWARE_SAFE_SELECTION", True)
    before = before if isinstance(before, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}
    base_analysis = base_analysis if isinstance(base_analysis, dict) else {}
    profile = profile if isinstance(profile, dict) else {}
    warnings = sorted(set(warnings or []))
    cand_lufs = _v57_analysis_metric(candidate, "integrated_lufs", "lufs", default=-99.0)
    base_lufs = _v57_analysis_metric(base_analysis, "integrated_lufs", "lufs", default=-99.0)
    cand_lra = _analysis_lra_lu(candidate)
    base_lra = _analysis_lra_lu(base_analysis)
    input_lra = _analysis_lra_lu(before)
    cand_crest = _v57_analysis_metric(candidate, "crest_factor_db", "crest_db", default=99.0)
    base_crest = _v57_analysis_metric(base_analysis, "crest_factor_db", "crest_db", default=cand_crest)
    input_crest = _v57_analysis_metric(before, "crest_factor_db", "crest_db", default=base_crest)
    lra_delta_input = None if (input_lra is None or cand_lra is None) else float(cand_lra) - float(input_lra)
    lra_delta_base = None if (base_lra is None or cand_lra is None) else float(cand_lra) - float(base_lra)
    crest_delta_input = None if (input_crest is None or cand_crest is None) else float(cand_crest) - float(input_crest)
    crest_delta_base = None if (base_crest is None or cand_crest is None) else float(cand_crest) - float(base_crest)
    hard_lra_floor = _env_float("BUSY_V57_HARD_LRA_FLOOR_LU", 1.05, 0.2, 4.0)
    severe_lra_floor = _env_float("BUSY_V57_SEVERE_LRA_FLOOR_LU", 0.90, 0.1, 3.0)
    max_lra_loss = _env_float("BUSY_V57_MAX_LRA_LOSS_LU", 2.25, 0.4, 8.0)
    max_crest_loss = _env_float("BUSY_V57_MAX_CREST_LOSS_DB", 4.0, 0.8, 10.0)
    hard_crest_floor = float(profile.get("hard_crest_min_db", 4.5) or 4.5)
    blockers: list[str] = []
    penalties: list[str] = []
    if cand_lra is not None:
        if float(cand_lra) < severe_lra_floor:
            blockers.append("v57_lra_severe_flattening")
        elif float(cand_lra) < hard_lra_floor and ("lra_too_flat" in warnings or "lra_reduced_too_much" in warnings):
            blockers.append("v57_lra_hard_floor")
        elif float(cand_lra) < hard_lra_floor:
            penalties.append("v57_lra_near_hard_floor")
    if lra_delta_input is not None and float(lra_delta_input) < -max_lra_loss:
        blockers.append("v57_lra_collapse_vs_input")
    if lra_delta_base is not None and float(lra_delta_base) < -max(0.75, max_lra_loss * 0.45):
        blockers.append("v57_lra_collapse_vs_base")
    if cand_crest is not None:
        if float(cand_crest) < hard_crest_floor:
            blockers.append("v57_transient_crest_hard_floor")
        elif float(cand_crest) < hard_crest_floor + 0.25:
            penalties.append("v57_transient_crest_near_floor")
    if "transient_loss_delta" in warnings:
        if crest_delta_input is not None and float(crest_delta_input) < -max_crest_loss:
            blockers.append("v57_transient_loss_vs_input")
        elif crest_delta_base is not None and float(crest_delta_base) < -max(1.25, max_crest_loss * 0.35):
            blockers.append("v57_transient_loss_vs_base")
        else:
            penalties.append("v57_transient_loss_warning")
    if ("lra_too_flat" in warnings or "lra_reduced_too_much" in warnings) and "transient_loss_delta" in warnings:
        blockers.append("v57_combined_lra_transient_damage")
    loudness_gain = float(cand_lufs or -99.0) - float(base_lufs or -99.0)
    # If a candidate still misses the floor, it does not earn the right to carry
    # severe damage just because it is louder than base.
    if loudness_gain < _env_float("BUSY_V57_DAMAGE_MIN_MEANINGFUL_GAIN_DB", 0.25, 0.0, 1.0) and blockers:
        blockers.append("v57_damage_without_meaningful_loudness_gain")
    reject = bool(enabled and blockers)
    penalty_score = len(set(penalties)) * 4.0 + len(set(blockers)) * 24.0
    return {
        "schema_version": "busy_v57_damage_aware_candidate_gate_v8_5_3_57_2",
        "enabled": bool(enabled),
        "active": bool(enabled),
        "accepted_by_v57": bool(not reject),
        "reject": bool(reject),
        "blockers": sorted(set(blockers)),
        "penalties": sorted(set(penalties)),
        "penalty_score": round(float(penalty_score), 3),
        "candidate_lufs": round(float(cand_lufs), 3) if cand_lufs is not None else None,
        "base_lufs": round(float(base_lufs), 3) if base_lufs is not None else None,
        "loudness_gain_vs_base_db": round(float(loudness_gain), 3),
        "commercial_floor_lufs": round(float(floor_lufs), 3),
        "target_lufs": round(float(target_lufs), 3),
        "candidate_lra_lu": round(float(cand_lra), 3) if cand_lra is not None else None,
        "base_lra_lu": round(float(base_lra), 3) if base_lra is not None else None,
        "input_lra_lu": round(float(input_lra), 3) if input_lra is not None else None,
        "lra_delta_vs_input_lu": round(float(lra_delta_input), 3) if lra_delta_input is not None else None,
        "lra_delta_vs_base_lu": round(float(lra_delta_base), 3) if lra_delta_base is not None else None,
        "candidate_crest_db": round(float(cand_crest), 3) if cand_crest is not None else None,
        "base_crest_db": round(float(base_crest), 3) if base_crest is not None else None,
        "input_crest_db": round(float(input_crest), 3) if input_crest is not None else None,
        "crest_delta_vs_input_db": round(float(crest_delta_input), 3) if crest_delta_input is not None else None,
        "crest_delta_vs_base_db": round(float(crest_delta_base), 3) if crest_delta_base is not None else None,
        "warnings_considered": warnings,
        "policy": "louder candidates are rejected when measured LRA/transient damage crosses hard gates; commercial floor is not allowed to override musical damage.",
    }


def _commercial_loudness_candidate_hard_blocks(
    before: dict[str, Any],
    after: dict[str, Any],
    decision: dict[str, Any],
    profile: dict[str, Any],
    warnings: list[str],
) -> list[str]:
    q_before = before.get("quality_indices", {}) if isinstance(before.get("quality_indices", {}), dict) else {}
    q_after = after.get("quality_indices", {}) if isinstance(after.get("quality_indices", {}), dict) else {}
    ms = after.get("ms", {}) if isinstance(after.get("ms", {}), dict) else {}
    hard: list[str] = []
    warning_set = set(warnings or [])
    experimental = bool(profile.get("adaptive_hot_commercial") or profile.get("experimental_hot_commercial"))
    hard_warning_names = [
        "true_peak_too_hot",
        "stereo_correlation_too_low",
        "low_end_too_wide",
        "mono_collapse_risk",
        "stereo_qc_low_end_side_excess",
        "side_high_shimmer_increased",
        "v7_shimmer_guard",
    ]
    if experimental:
        # In experimental mode side shimmer is allowed as a warning after the
        # direct residue light-control stage; true peak, mono/correlation and
        # severe low-end failures remain hard stops.
        hard_warning_names = [x for x in hard_warning_names if x not in {"side_high_shimmer_increased", "v7_shimmer_guard"}]
    for name in hard_warning_names:
        if name in warning_set:
            hard.append(name)
    crest = float(after.get("crest_factor_db", 99.0) or 99.0)
    if crest < float(profile.get("hard_crest_min_db", 5.0) or 5.0):
        hard.append("commercial_crest_hard_floor")
    corr = float(after.get("correlation", 1.0) or 1.0)
    if corr < max(0.03, float((decision.get("targets", {}) or {}).get("correlation_floor", 0.05) or 0.05) - 0.02):
        hard.append("commercial_correlation_floor")
    low_side = float(ms.get("side_ratio_lowband", 0.0) or 0.0)
    low_side_max = float((decision.get("targets", {}) or {}).get("low_end_side_ratio_max", 0.14) or 0.14)
    low_side_hard = max(0.18 if experimental else 0.14, low_side_max + (0.07 if experimental else 0.04))
    if low_side > low_side_hard:
        hard.append("commercial_low_side_floor")
    harsh_delta = float(q_after.get("harshness_index_db", 0.0) or 0.0) - float(q_before.get("harshness_index_db", 0.0) or 0.0)
    harsh_limit = float(profile.get("harshness_hard_max_db", 1.25) or 1.25)
    if harsh_delta > harsh_limit:
        if experimental:
            # In adaptive commercial mode, harshness/side-fizz is a recorded
            # warning for the user and QC ledger, not an automatic rejection.
            # True peak, crest, correlation and low-side failures still block.
            pass
        else:
            hard.append("commercial_harshness_hard_delta")
    if ("lra_too_flat" in warning_set or "lra_reduced_too_much" in warning_set) and not profile.get("allow_moderate_lra_warning"):
        # LRA warnings are not always fatal for hot commercial masters. In
        # Commercial/Maximum they are score penalties unless the measured LRA
        # crosses the family hard floor and the candidate is also near the crest floor.
        policy = profile.get("commercial_tolerance_policy", {}) if isinstance(profile.get("commercial_tolerance_policy", {}), dict) else {}
        preset = str(profile.get("mastering_intensity") or policy.get("intensity_preset") or _mastering_intensity_key())
        if preset not in {"commercial", "maximum"}:
            if crest < float(profile.get("crest_min_db", 5.6) or 5.6) + 0.35:
                hard.append("commercial_lra_flat_with_low_crest")
    return sorted(set(hard))


def _candidate_problem_set(
    before: dict[str, Any],
    after: dict[str, Any],
    decision: dict[str, Any],
    profile: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    base_problem_set: list[str] | None = None,
) -> dict[str, Any]:
    """Build a strict, relative blocker/problem set for hot commercial candidates.

    This is deliberately not a one-off threshold patch.  It merges the existing
    safety, LRA, A/B delta, codec and playback checks into a single unresolved
    blocker set so a candidate can only win if it is both louder and objectively
    cleaner than the current safe reference.
    """
    profile = profile if isinstance(profile, dict) else {}
    warnings = sorted(set(warnings or []))
    safety = _safety_risk(before, after, decision)
    ab = _ab_delta_qc(before, after)
    lra = _lra_guard_summary(before, after, decision)
    codec = _codec_preview_guard(before, after)
    playback = _playback_translation_report(before, after)
    hard = _commercial_loudness_candidate_hard_blocks(before, after, decision, profile, sorted(set(warnings + safety)))
    problem_set = sorted(set(
        list(warnings)
        + list(safety or [])
        + list(ab.get("warnings", []) or [])
        + list(lra.get("warnings", []) or [])
        + list(codec.get("warnings", []) or [])
        + list(playback.get("warnings", []) or [])
        + list(hard or [])
    ))
    base_set = set(base_problem_set or [])
    new_set = sorted([p for p in problem_set if p not in base_set])
    critical_names = {
        "true_peak_too_hot",
        "commercial_crest_hard_floor",
        "commercial_lra_flat_with_low_crest",
        "commercial_overpush",
        "crest_factor_too_low",
        "transient_loss_delta",
        "lra_too_flat",
        "lra_reduced_too_much",
        "codec_density_distortion_risk",
        "stereo_correlation_too_low",
        "mono_collapse_risk",
        "commercial_low_side_floor",
    }
    # v8.5.3.29: in commercial-reference match mode, dense EDM/pop masters
    # may legitimately trade crest/LRA/transient metrics for loudness as long as
    # they do not cross hard failure floors. Keep these visible as warnings and
    # score penalties, but do not make them automatic new-critical blockers.
    if profile.get("commercial_reference_match"):
        critical_names.difference_update({
            "crest_factor_too_low",
            "transient_loss_delta",
            "lra_too_flat",
            "lra_reduced_too_much",
        })
    critical = sorted([p for p in problem_set if p in critical_names or str(p).startswith("commercial_")])
    critical_new = sorted([p for p in new_set if p in critical_names or str(p).startswith("commercial_")])
    metrics = _risk_metrics(before, after)
    damage_score = 0.0
    # Damage score is relative and multi-factor: the goal is not to ban a single
    # number but to reject candidates whose total new damage outweighs loudness.
    damage_score += max(0.0, -float(metrics.get("lra_delta_lu", 0.0))) * 0.75
    damage_score += max(0.0, float((before or {}).get("crest_factor_db", 99.0) or 99.0) - float((after or {}).get("crest_factor_db", 99.0) or 99.0)) * 0.85
    damage_score += max(0.0, float(metrics.get("harshness_delta_db", 0.0))) * 1.15
    damage_score += max(0.0, float(metrics.get("air_delta_db", 0.0)) - 0.75) * 0.65
    damage_score += len(new_set) * 1.35
    damage_score += len(critical_new) * 4.0
    return {
        "schema_version": "busy_candidate_problem_set_v8_5_3_21",
        "policy": "strict multi-blocker set; louder candidates cannot pass with new unresolved musical damage",
        "problem_set": problem_set,
        "base_problem_set": sorted(base_set),
        "new_vs_base": new_set,
        "critical_problem_set": critical,
        "critical_new_vs_base": critical_new,
        "damage_score": round(float(damage_score), 3),
        "risk_metrics": {k: round(float(v), 4) for k, v in metrics.items() if isinstance(v, (int, float))},
        "ab_delta_qc": ab,
        "lra_guard": lra,
        "codec_preview_guard": codec,
        "playback_translation": {"overall_score": playback.get("overall_score"), "warnings": playback.get("warnings", [])},
    }


def _candidate_acceptance_decision(
    before: dict[str, Any],
    candidate_analysis: dict[str, Any],
    decision: dict[str, Any],
    profile: dict[str, Any],
    warnings: list[str] | None,
    base_analysis: dict[str, Any],
    base_problem_set: list[str] | None,
    floor_lufs: float,
    target_lufs: float,
) -> dict[str, Any]:
    cset = _candidate_problem_set(before, candidate_analysis, decision, profile, warnings or [], base_problem_set=base_problem_set)
    cand_lufs = float(candidate_analysis.get("integrated_lufs", -99.0) or -99.0)
    base_lufs = float(base_analysis.get("integrated_lufs", -99.0) or -99.0)
    loudness_gain = cand_lufs - base_lufs
    unresolved = list(cset.get("critical_new_vs_base", []) or [])
    # Candidates with any newly introduced severe blocker are not warnings; they
    # are failed attempts and the solver must choose a lower-push or cleaner bundle.
    accepted = bool(loudness_gain > 0.18 and not unresolved)
    damage_threshold = max(6.5, 2.2 + max(0.0, loudness_gain) * 1.35)
    if profile.get("commercial_reference_match"):
        damage_threshold = max(8.5, 3.0 + max(0.0, loudness_gain) * 1.65)
    if accepted and cset.get("damage_score", 999.0) > damage_threshold:
        accepted = False
        unresolved.append("damage_score_exceeds_loudness_benefit")
    if accepted and cand_lufs >= float(floor_lufs) and cset.get("critical_problem_set"):
        # A hot candidate is not allowed to hit the floor by creating a new severe
        # problem set.  If the severe problems existed already, lower-push retry
        # can still compare, but it must not create extra ones.
        pass
    v57_gate = _v57_damage_aware_candidate_gate(
        before,
        candidate_analysis,
        base_analysis,
        decision,
        profile,
        warnings or [],
        floor_lufs,
        target_lufs,
    )
    if v57_gate.get("reject"):
        accepted = False
        unresolved.extend(v57_gate.get("blockers", []) or [])
    return {
        "schema_version": "busy_candidate_acceptance_v8_5_3_57_damage_aware_safe_selection",
        "accepted": bool(accepted),
        "loudness_gain_vs_base_db": round(float(loudness_gain), 3),
        "candidate_lufs": round(float(cand_lufs), 3),
        "base_lufs": round(float(base_lufs), 3),
        "reached_floor": bool(cand_lufs >= float(floor_lufs)),
        "reached_target": bool(cand_lufs >= float(target_lufs) - 0.20),
        "unresolved_blocker_set": sorted(set(unresolved)),
        "problem_set": cset,
        "v57_damage_aware_candidate_gate": v57_gate,
        "policy": "accept only if louder and no new critical blocker/damage set remains; v57 rejects loud candidates that cross LRA/transient damage gates",
    }


def _extract_agent_strategy_opinions(decision: dict[str, Any]) -> dict[str, Any]:
    """Collect existing GPT-mini/Gemini/Realtime opinions without pretending GPT can hear audio."""
    decision = decision if isinstance(decision, dict) else {}
    final_ctx = decision.get("final_decision_context", {}) if isinstance(decision.get("final_decision_context", {}), dict) else {}
    final_arb = decision.get("final_arbitration", {}) if isinstance(decision.get("final_arbitration", {}), dict) else {}
    validation = final_ctx.get("post_decision_audio_validation") or final_arb.get("post_decision_audio_validation") or {}
    if not isinstance(validation, dict):
        validation = {}
    return {
        "schema_version": "busy_agent_strategy_opinions_v8_5_3_21",
        "gpt_mini_text_opinion": {
            "role": "text_only_first_opinion",
            "model": decision.get("model_used"),
            "selected_profile": decision.get("selected_profile") or decision.get("primary_profile"),
            "genre_mix": decision.get("genre_mix"),
            "role_first_contract": final_ctx.get("role_first_dsp_contract", {}),
            "notes": decision.get("notes", [])[:8] if isinstance(decision.get("notes", []), list) else [],
            "important_limitation": "GPT-mini does not receive audio; it reasons from text metrics, genre_evidence, blocker maps, and other judges.",
        },
        "gemini_audio_opinion": {
            "role": "audio_based_opinion_when_available",
            "verdict": ((validation.get("verdicts") or {}) if isinstance(validation.get("verdicts"), dict) else {}).get("gemini_flash_lite"),
            "support_score": ((validation.get("support_scores") or {}) if isinstance(validation.get("support_scores"), dict) else {}).get("gemini_flash_lite"),
            "missed_roles": [x for x in (validation.get("missed_roles", []) or []) if isinstance(x, dict) and x.get("validator") == "gemini_flash_lite"],
        },
        "realtime_audio_opinion": {
            "role": "audio_based_opinion_when_available",
            "verdict": ((validation.get("verdicts") or {}) if isinstance(validation.get("verdicts"), dict) else {}).get("openai_realtime_mini"),
            "support_score": ((validation.get("support_scores") or {}) if isinstance(validation.get("support_scores"), dict) else {}).get("openai_realtime_mini"),
            "missed_roles": [x for x in (validation.get("missed_roles", []) or []) if isinstance(x, dict) and x.get("validator") == "openai_realtime_mini"],
        },
        "council_validation_summary": validation,
    }


def _build_multi_agent_virtual_premaster_strategy_council(
    before: dict[str, Any],
    base_analysis: dict[str, Any],
    decision: dict[str, Any],
    virtual_hot_solver: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    ge = decision.get("genre_evidence", {}) if isinstance(decision.get("genre_evidence", {}), dict) else {}
    metrics = ge.get("metrics", {}) if isinstance(ge.get("metrics", {}), dict) else {}
    return {
        "schema_version": "busy_multi_agent_virtual_premaster_strategy_council_v8_5_3_21",
        "enabled": True,
        "primary_evidence_layer": {
            "name": "genre_evidence",
            "schema_version": ge.get("schema_version"),
            "metrics": {k: metrics.get(k) for k in [
                "tempo_bpm", "transient_density_per_sec", "four_on_floor_confidence",
                "side_ratio_lowband", "side_ratio_8k_14k", "correlation_avg",
                "tf_voice_probability", "tf_synth_probability", "tf_electronic_probability",
                "tf_danceable_probability", "edm_instrument_score", "rock_instrument_score",
            ] if k in metrics},
            "evidence_atoms": ge.get("evidence_atoms", {}),
            "audio_group_scores": ge.get("audio_group_scores", {}),
            "hybrid_role_evidence": ge.get("hybrid_role_evidence", {}),
            "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        },
        "gpt_mini_and_audio_model_opinions": _extract_agent_strategy_opinions(decision),
        "virtual_pressure_test": {
            "representative_section_map": virtual_hot_solver.get("representative_section_map", {}),
            "virtual_limiter_sweep": virtual_hot_solver.get("virtual_limiter_sweep", {}),
            "blocker_capacity_map": virtual_hot_solver.get("blocker_capacity_map", {}),
        },
        "multi_blocker_solution_bundle": virtual_hot_solver.get("blocker_solution_plan", {}),
        "premaster_strategy": {
            "basis": "primary evidence + GPT-mini text opinion + Gemini/Realtime audio opinions + pressure-test blocker set",
            "final_push_strategy": virtual_hot_solver.get("final_push_strategy", {}),
            "policy": "solve all blockers appearing at the same gain step as a bundle, then push; unresolved new problems disqualify the candidate instead of passing as warnings",
        },
        "final_arbitration_policy": {
            "gpt_mini_role": "first text opinion, not final authority",
            "audio_judge_role": "audio-based role/problem/strategy opinion when available",
            "high_reasoning_role": "final arbitration when available through existing high-accuracy route; deterministic QC veto always applies",
            "python_qc_role": "final veto on unresolved problem sets and selected-candidate/final-output mismatch",
        },
    }



def _condition_for_commercial_push(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any],
    current_analysis: dict[str, Any],
    push_db: float,
    profile: dict[str, Any],
    virtual_hot_plan: dict[str, Any] | None = None,
    resolution_plan: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Control likely limiter failure points before adding more level."""
    out = y
    try:
        _commercial_condition_duration_sec = float(np.asarray(out).shape[0] / max(int(sr), 1))
    except Exception:
        _commercial_condition_duration_sec = 0.0
    _commercial_condition_fast = bool(
        _env_bool("BUSY_COMMERCIAL_CONDITION_LIGHTWEIGHT", True)
        and _commercial_condition_duration_sec >= _env_float("BUSY_COMMERCIAL_CONDITION_LIGHTWEIGHT_MIN_SEC", 12.0, 3.0, 120.0)
    )
    q_before = before.get("quality_indices", {}) if isinstance(before.get("quality_indices", {}), dict) else {}
    q_cur = current_analysis.get("quality_indices", {}) if isinstance(current_analysis.get("quality_indices", {}), dict) else {}
    ms = current_analysis.get("ms", {}) if isinstance(current_analysis.get("ms", {}), dict) else {}
    low_side = float(ms.get("side_ratio_lowband", 0.0) or 0.0)
    high_side = float(ms.get("side_ratio_8k_14k", 0.0) or 0.0)
    harsh_delta = float(q_cur.get("harshness_index_db", 0.0) or 0.0) - float(q_before.get("harshness_index_db", 0.0) or 0.0)
    air_delta = float(q_cur.get("air_index_db", 0.0) or 0.0) - float(q_before.get("air_index_db", 0.0) or 0.0)
    crest = float(current_analysis.get("crest_factor_db", 99.0) or 99.0)
    intensity = float(np.clip(0.65 + max(0.0, push_db) * 0.20, 0.65, 1.18))
    moves: list[dict[str, Any]] = []
    dyn: list[dict[str, Any]] = []
    notes: list[str] = []
    resolution_plan = resolution_plan if isinstance(resolution_plan, dict) else {}
    resolution_bundle = resolution_plan.get("chosen_bundle", {}) if isinstance(resolution_plan.get("chosen_bundle", {}), dict) else {}

    if _commercial_condition_fast and _env_bool("BUSY_COMMERCIAL_CONDITION_EARLY_LIGHTWEIGHT_RETURN", True):
        # v8.5.3.62.8.2 runtime guard: commercial finish is a late candidate
        # search, not the core mastering path.  On full-length tracks the richer
        # blocker-conditioner can dominate wall time after the standard final is
        # already safe.  Use a bounded micro-clip-only preconditioner; the real
        # hot render, QC, selected-master guard and export lock still verify the
        # actual output.
        clip_drive = float(np.clip(0.30 + max(0.0, push_db) * 0.20, 0.20, 0.90))
        clip_mix = float(np.clip(0.055 + max(0.0, push_db) * 0.020, 0.040, 0.140))
        out = soft_clip(out, drive_db=clip_drive, mix=clip_mix)
        notes.append("runtime_early_lightweight_commercial_condition_microclip_only")
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), {
            "notes": notes,
            "runtime_lightweight_condition": True,
            "early_lightweight_return": True,
            "condition_duration_sec": round(float(_commercial_condition_duration_sec), 3),
            "static_move_count": 0,
            "dynamic_move_count": 0,
            "micro_clip_drive_db": round(clip_drive, 3),
            "micro_clip_mix": round(clip_mix, 4),
            "virtual_mix_conditioning_render": {},
            "virtual_mix_conditioning_seven_stage_receipt": {},
            "research_driven_blocker_solution_planner": resolution_plan if resolution_plan else {},
            "v55_controlled_direct_activation": {},
            "policy": "late commercial candidate conditioner is bounded for Cloud Run; final audio-domain QC remains authoritative",
        }

    if resolution_bundle:
        rb_low = float(resolution_bundle.get("low_side_mono_amount", 0.0) or 0.0)
        if rb_low > 0.0:
            out = ms_low_mono(out, sr, cutoff_hz=145.0, amount=float(np.clip(rb_low, 0.0, 0.92)))
            notes.append("research_planner_low_side_mono_before_push")
        rb_static = [m for m in (resolution_bundle.get("static_moves", []) or []) if isinstance(m, dict)]
        rb_dynamic = [m for m in (resolution_bundle.get("dynamic_moves", []) or []) if isinstance(m, dict)]
        if rb_static:
            moves.extend(rb_static)
            notes.append("research_planner_static_moves_before_push")
        if rb_dynamic:
            dyn.extend(rb_dynamic)
            notes.append("research_planner_dynamic_moves_before_push")
        rb_transient = float(resolution_bundle.get("transient_micro_expander_amount", 0.0) or 0.0)
        if rb_transient > 0.0:
            out = transient_micro_expander(out, sr, amount=float(np.clip(rb_transient, 0.0, 0.075)))
            notes.append("research_planner_transient_micro_protection_before_push")

    if low_side > 0.075:
        out = ms_low_mono(out, sr, cutoff_hz=130.0, amount=float(np.clip(0.42 + low_side * 2.0, 0.45, 0.78)))
        moves.append({"type": "bell", "freq": 115, "gain_db": -0.18 * intensity, "q": 0.9, "target": "side"})
        dyn.append({"type": "bell", "freq": 95, "gain_db": -0.24 * intensity, "q": 0.85, "target": "side"})
        notes.append("low_side_control_before_push")
    if harsh_delta > 0.45:
        dyn.append({"type": "bell", "freq": 4300, "gain_db": -0.20 * intensity, "q": 1.35, "target": "mid"})
        notes.append("upper_mid_control_before_push")
    if high_side > 0.50 or air_delta > 0.9:
        dyn.append({"type": "bell", "freq": 8800, "gain_db": -0.26 * intensity, "q": 1.25, "target": "side"})
        moves.append({"type": "high_shelf", "freq": 11800, "gain_db": -0.12 * intensity, "q": 0.7, "target": "side"})
        notes.append("side_high_control_before_push")

    # v8.5.3.16: the hot-commercial path is no longer a blind fixed push.
    # If the virtual solver predicted blocker-specific correction moves, apply
    # those upstream before adding level.  This keeps the actual full render close
    # to the calculated final push strategy.
    virtual_hot_plan = virtual_hot_plan if isinstance(virtual_hot_plan, dict) else {}
    solution = virtual_hot_plan.get("blocker_solution_plan", {}) if isinstance(virtual_hot_plan.get("blocker_solution_plan", {}), dict) else {}
    if solution:
        vh_low = float(solution.get("low_side_mono_amount", 0.0) or 0.0)
        if vh_low > 0.0:
            out = ms_low_mono(out, sr, cutoff_hz=145.0, amount=float(np.clip(vh_low, 0.0, 0.90)))
            notes.append("virtual_hot_low_side_mono_before_push")
        vh_static = [m for m in (solution.get("static_moves", []) or []) if isinstance(m, dict)]
        vh_dynamic = [m for m in (solution.get("dynamic_moves", []) or []) if isinstance(m, dict)]
        if vh_static:
            moves.extend(vh_static)
            notes.append("virtual_hot_static_moves_before_push")
        if vh_dynamic:
            dyn.extend(vh_dynamic)
            notes.append("virtual_hot_dynamic_moves_before_push")

    if bool(profile.get("adaptive_hot_commercial")) and _env_bool("BUSY_ADAPTIVE_RESIDUE_CONTROL", True) and not solution:
        # Fallback when virtual solver is disabled/unavailable: if the full-rate
        # centrifuge saw high-band/ultra-air residue, reduce the exposed band
        # before the gain push rather than rejecting all louder candidates.
        c = _extract_frequency_band_centrifuge_context(before, decision=None)
        gs = c.get("global_summary", {}) if isinstance(c.get("global_summary", {}), dict) else {}
        high_risk = float(gs.get("high_band_residue_risk", 0.0) or 0.0)
        if high_risk >= _env_float("BUSY_ADAPTIVE_RESIDUE_MIN_CONF", _env_float("BUSY_AI_RESIDUE_DIRECT_MIN_CONF", 0.45, 0.25, 0.95), 0.25, 0.95):
            exp_strength = _env_float("BUSY_ADAPTIVE_RESIDUE_STRENGTH", _env_float("BUSY_AI_RESIDUE_DIRECT_STRENGTH", 1.0, 0.20, 2.0), 0.20, 2.0)
            dyn.append({"type": "bell", "freq": 14500, "gain_db": -0.34 * intensity * exp_strength, "q": 0.95, "target": "stereo", "source": "candidate_adaptive_air_control"})
            dyn.append({"type": "bell", "freq": 19000, "gain_db": -0.24 * intensity * exp_strength, "q": 0.85, "target": "stereo", "source": "candidate_adaptive_ultra_air_control"})
            notes.append("adaptive_residue_control_before_push_fallback")
    if crest < float(profile.get("crest_min_db", 5.6) or 5.6) + 0.7:
        # Do not use a full transient restore; just give the limiter a little edge
        # to avoid a denser-but-smaller feeling result.
        out = transient_micro_expander(out, sr, amount=0.018)
        notes.append("micro_transient_protection_before_push")
    if _env_bool("BUSY_COMMERCIAL_CONDITION_FAST_MOVE_CAP", True):
        _max_static = int(_env_float("BUSY_COMMERCIAL_CONDITION_MAX_STATIC_MOVES", 4 if _commercial_condition_fast else 6, 1, 16))
        _max_dynamic = int(_env_float("BUSY_COMMERCIAL_CONDITION_MAX_DYNAMIC_MOVES", 0 if _commercial_condition_fast else 6, 0, 16))
        if len(moves) > _max_static:
            notes.append("runtime_static_move_cap_applied")
            moves = moves[:_max_static]
        if _commercial_condition_fast and dyn:
            notes.append("runtime_dynamic_eq_skipped_in_lightweight_commercial_condition")
            dyn = []
        elif len(dyn) > _max_dynamic:
            notes.append("runtime_dynamic_move_cap_applied")
            dyn = dyn[:_max_dynamic]
    if moves:
        out = apply_eq_moves(out, sr, moves, scale=1.0)
    if dyn:
        out = dynamic_eq(out, sr, dyn, intensity=1.0)

    # Micro-clip peaks before the limiter, so the next gain push raises density
    # rather than only slamming the final lookahead limiter.
    experimental = bool(profile.get("adaptive_hot_commercial") or profile.get("experimental_hot_commercial"))
    if experimental:
        # Stronger density prep for A/B testing: this is where the hot path
        # actually gains LUFS instead of only hitting the limiter ceiling.
        mb_amount = float(np.clip(0.28 + max(0.0, push_db) * 0.08, 0.28, 0.62))
        if resolution_bundle:
            mb_amount *= float(np.clip(float(resolution_bundle.get("multiband_amount_scale", 1.0) or 1.0), 0.35, 1.15))
            v55_lift = float(np.clip(float(resolution_bundle.get("v55_density_lift_scale", 1.0) or 1.0), 0.80, 1.20))
            if abs(v55_lift - 1.0) > 1e-6:
                mb_amount *= v55_lift
                notes.append("v55_controlled_density_lift_before_push")
            notes.append("research_planner_multiband_density_responsibility_adjust")
        if _commercial_condition_fast and _env_bool("BUSY_COMMERCIAL_CONDITION_SKIP_HOT_MULTIBAND", True):
            notes.append("runtime_hot_multiband_density_skipped_in_lightweight_commercial_condition")
        else:
            out = multiband_compress(out, sr, mode="hot", amount=float(np.clip(mb_amount, 0.10, 0.68)))
            notes.append("experimental_hot_multiband_density_before_push")
        clip_drive = float(np.clip(0.34 + max(0.0, push_db) * 0.28, 0.30, 1.55))
        clip_mix = float(np.clip(0.070 + max(0.0, push_db) * 0.030, 0.070, 0.235))
    else:
        clip_drive = float(np.clip(0.14 + max(0.0, push_db) * 0.16, 0.12, 0.52))
        clip_mix = float(np.clip(0.035 + max(0.0, push_db) * 0.018, 0.035, 0.085))
    if solution:
        clip_drive = float(np.clip(clip_drive + float(solution.get("clip_drive_delta_db", 0.0) or 0.0), 0.10, 2.10))
        clip_mix = float(np.clip(clip_mix + float(solution.get("clip_mix_delta", 0.0) or 0.0), 0.025, 0.32))
        if solution.get("clip_drive_delta_db") or solution.get("clip_mix_delta"):
            notes.append("virtual_hot_clip_limit_distribution_adjust")
    if resolution_bundle:
        clip_drive = float(np.clip(clip_drive + float(resolution_bundle.get("clip_drive_delta_db", 0.0) or 0.0), 0.05, 2.10))
        clip_mix = float(np.clip(clip_mix + float(resolution_bundle.get("clip_mix_delta", 0.0) or 0.0), 0.015, 0.32))
        clip_drive *= float(np.clip(float(resolution_bundle.get("clip_drive_scale", 1.0) or 1.0), 0.35, 1.10))
        clip_mix *= float(np.clip(float(resolution_bundle.get("clip_mix_scale", 1.0) or 1.0), 0.30, 1.10))
        clip_drive = float(np.clip(clip_drive, 0.05, 2.10))
        clip_mix = float(np.clip(clip_mix, 0.012, 0.32))
        notes.append("research_planner_clip_limit_responsibility_adjust")
    out = soft_clip(out, drive_db=clip_drive, mix=clip_mix)
    notes.append("experimental_hot_clip_before_commercial_push" if experimental else "micro_clip_before_commercial_push")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), {
        "notes": notes,
        "runtime_lightweight_condition": bool(_commercial_condition_fast),
        "condition_duration_sec": round(float(_commercial_condition_duration_sec), 3),
        "static_move_count": len(moves),
        "dynamic_move_count": len(dyn),
        "micro_clip_drive_db": round(clip_drive, 3),
        "micro_clip_mix": round(clip_mix, 4),
        "virtual_mix_conditioning_render": resolution_bundle.get("virtual_mix_conditioning_render", {}) if isinstance(resolution_bundle, dict) else {},
        "virtual_mix_conditioning_seven_stage_receipt": resolution_bundle.get("virtual_mix_conditioning_seven_stage_receipt", {}) if isinstance(resolution_bundle, dict) else {},
        "research_driven_blocker_solution_planner": resolution_plan if resolution_plan else {},
        "v55_controlled_direct_activation": resolution_bundle.get("v55_ivrcf_controlled_direct_activation", {}) if isinstance(resolution_bundle, dict) else {},
    }




# ---------------------------------------------------------------------------
# v8.5.3.37: adaptive final-limited preview-master audio review helpers.
# ---------------------------------------------------------------------------

def _preview_master_segment_candidates(duration: float, before: dict[str, Any], seg_len: float) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if duration <= seg_len + 0.2:
        return [("full_track", 0.0)]
    out.append(("middle_groove", min(max(0.0, duration * 0.45), max(0.0, duration - seg_len))))
    summary = ((before.get("segment_analysis") or {}).get("summary") or {}) if isinstance(before.get("segment_analysis"), dict) else {}
    for key in ["loudest_10s", "brightest_10s", "most_bass_heavy_10s", "most_compressed_section"]:
        sec = summary.get(key, {}) if isinstance(summary, dict) else {}
        if isinstance(sec, dict) and sec.get("start_sec") is not None:
            try:
                s = float(sec.get("start_sec") or 0.0)
            except Exception:
                s = 0.0
            out.append((key, min(max(0.0, s + 1.5), max(0.0, duration - seg_len))))
    out.append(("late_peak_candidate", min(max(0.0, duration * 0.72), max(0.0, duration - seg_len))))
    return out


def _build_preview_master_source_audio(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any],
    *,
    total_sec: float = 28.0,
    segment_sec: float = 7.0,
    max_segments: int = 4,
    adaptive_plan: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
    else:
        arr2 = arr
    duration = float(arr2.shape[0] / max(int(sr), 1))
    plan = adaptive_plan if isinstance(adaptive_plan, dict) else {}
    if plan.get("enabled"):
        total_sec = float(plan.get("target_total_sec", total_sec) or total_sec)
        segment_sec = float(plan.get("segment_sec", segment_sec) or segment_sec)
        max_segments = int(plan.get("segment_count", max_segments) or max_segments)
    seg_len = float(max(3.0, min(segment_sec, max(3.0, total_sec))))
    target_segments = int(max(1, min(max_segments, round(total_sec / max(seg_len, 1e-6)))))
    candidates = _preview_master_segment_candidates(duration, before if isinstance(before, dict) else {}, seg_len)
    selected: list[tuple[str, float, float]] = []
    min_distance = max(2.0, seg_len * 0.65)
    for label, start in candidates:
        if len(selected) >= target_segments:
            break
        s0 = min(max(0.0, float(start)), max(0.0, duration - seg_len))
        if all(abs(s0 - prev) >= min_distance for _, prev, _ in selected):
            selected.append((label, s0, seg_len))
    i = 0
    while len(selected) < target_segments and duration > seg_len and i < 12:
        frac = (len(selected) + 1) / (target_segments + 1)
        s0 = min(max(0.0, duration * frac), max(0.0, duration - seg_len))
        if all(abs(s0 - prev) >= 1.0 for _, prev, _ in selected):
            selected.append((f"fallback_{i}", s0, seg_len))
        i += 1
    clips: list[np.ndarray] = []
    actual: list[dict[str, Any]] = []
    fade_n = int(max(1, min(round(sr * 0.025), round(seg_len * sr / 4))))
    for label, start, length in selected:
        a = int(round(start * sr))
        b = int(round(min(duration, start + length) * sr))
        clip = np.array(arr2[max(0, a):max(0, b)], dtype=np.float32, copy=True)
        if clip.size == 0:
            continue
        if clip.shape[0] > fade_n * 2:
            ramp = np.linspace(0.0, 1.0, fade_n, dtype=np.float32).reshape(-1, 1)
            clip[:fade_n] *= ramp
            clip[-fade_n:] *= ramp[::-1]
        clips.append(clip)
        actual.append({"label": label, "start_sec": round(float(start), 3), "duration_sec": round(float(clip.shape[0] / max(sr, 1)), 3)})
    if not clips:
        raise ValueError("preview_master_source_has_no_segments")
    silence = np.zeros((int(round(sr * 0.12)), arr2.shape[1]), dtype=np.float32)
    pieces: list[np.ndarray] = []
    for idx, clip in enumerate(clips):
        if idx > 0:
            pieces.append(silence)
        pieces.append(clip)
    preview = np.concatenate(pieces, axis=0).astype(np.float32, copy=False)
    if y.ndim == 1:
        preview_out = preview[:, 0]
    else:
        preview_out = preview
    return preview_out, {
        "schema_version": "busy_preview_master_source_windows_v8_5_3_37_adaptive",
        "total_audio_sec": round(float(preview.shape[0] / max(sr, 1)), 3),
        "source_duration_sec": round(float(duration), 3),
        "segment_sec": round(float(seg_len), 3),
        "segments": actual,
        "adaptive_preview_plan": plan if plan else {"enabled": False},
        "policy": "adaptive 20-40s representative windows only; this source preview is rendered through the candidate chain and final limiter for Gemini before full-track render",
    }


def _encode_preview_master_wav_base64(y: np.ndarray, sr: int, *, target_sr: int = 24000) -> dict[str, Any]:
    if sf is None:
        raise RuntimeError("soundfile_unavailable_for_preview_master_encoding")
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
    else:
        arr2 = arr
    if int(sr) != int(target_sr):
        try:
            g = int(np.gcd(int(sr), int(target_sr)))
            up = int(target_sr) // g
            down = int(sr) // g
            arr2 = resample_poly(arr2, up, down, axis=0).astype(np.float32, copy=False)
        except Exception:
            # Last resort: keep original SR rather than failing the master.
            target_sr = int(sr)
    peak = float(np.max(np.abs(arr2))) if arr2.size else 0.0
    if peak > 0.995:
        arr2 = arr2 * (0.96 / max(peak, 1e-9))
    bio = io.BytesIO()
    sf.write(bio, np.clip(arr2, -1.0, 1.0), int(target_sr), format="WAV", subtype="PCM_16")
    wav_bytes = bio.getvalue()
    return {
        "schema_version": "busy_preview_master_wav_base64_v8_5_3_37_adaptive",
        "format": "wav_pcm16",
        "sample_rate_hz": int(target_sr),
        "channels": int(arr2.shape[1]) if arr2.ndim == 2 else 1,
        "total_audio_sec": round(float(arr2.shape[0] / max(int(target_sr), 1)), 3),
        "peak_before_encode": round(float(peak), 6),
        "bytes": len(wav_bytes),
        "base64": base64.b64encode(wav_bytes).decode("ascii"),
    }



# ---------------------------------------------------------------------------
# v8.5.3.55.3.1: final listener review provenance binding.
# ---------------------------------------------------------------------------

def _json_compact_clone(value: Any, *, max_list: int = 24, max_dict: int = 80, max_text: int = 500) -> Any:
    """Return a JSON-safe compact clone suitable for stage_state/Supabase provenance."""
    try:
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value[:max_text]
        if isinstance(value, (list, tuple)):
            return [_json_compact_clone(v, max_list=max_list, max_dict=max_dict, max_text=max_text) for v in list(value)[:max_list]]
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for i, (k, v) in enumerate(value.items()):
                if i >= max_dict:
                    out["_truncated_keys"] = max(0, len(value) - max_dict)
                    break
                out[str(k)[:120]] = _json_compact_clone(v, max_list=max_list, max_dict=max_dict, max_text=max_text)
            return out
        return str(value)[:max_text]
    except Exception:
        return None


def _safe_hash_short(value: Any) -> str:
    try:
        blob = json.dumps(_json_compact_clone(value), sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
        return hashlib.sha1(blob).hexdigest()[:16]
    except Exception:
        return "unavailable"


def _first_dict(*values: Any) -> dict[str, Any]:
    for v in values:
        if isinstance(v, dict):
            return v
    return {}


def _last_dict_from_list(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        for v in reversed(value):
            if isinstance(v, dict):
                return v
    return {}


def _compact_move(move: Any) -> dict[str, Any]:
    if not isinstance(move, dict):
        return {}
    return {
        "type": move.get("type"),
        "target": move.get("target"),
        "freq": move.get("freq", move.get("frequency_hz")),
        "q": move.get("q"),
        "gain_db": move.get("gain_db", move.get("amount_db")),
        "source": move.get("source"),
        "module_id": move.get("module_id"),
        "v54_1_duplicate_axis_merge": move.get("v54_1_duplicate_axis_merge"),
        "v54_1_merged_sources": move.get("v54_1_merged_sources"),
        "v55_controlled_direct_activation": move.get("v55_controlled_direct_activation"),
    }


def _build_final_listener_lineage_provenance(
    *,
    commercial_finish_report: dict[str, Any],
    final_export_lock: dict[str, Any],
    final_analysis: dict[str, Any],
    decision: dict[str, Any],
) -> dict[str, Any]:
    """Bind Gemini final-listener review to the correction lineage that produced the delivered master.

    Gemini only hears the final highlight, but learning needs to know which
    module/candidate/export path produced that audible result.  This compact
    provenance pack is stored with the review and later used only as weak
    listener-priority context; it never authorizes direct DSP without current
    track evidence.
    """
    cfr = commercial_finish_report if isinstance(commercial_finish_report, dict) else {}
    fel = final_export_lock if isinstance(final_export_lock, dict) else {}
    plan = cfr.get("one_shot_candidate_plan") if isinstance(cfr.get("one_shot_candidate_plan"), dict) else {}
    vmcr = cfr.get("virtual_mix_conditioning_render") if isinstance(cfr.get("virtual_mix_conditioning_render"), dict) else {}
    render_attempt = _last_dict_from_list(cfr.get("render_attempts"))
    conditioning = render_attempt.get("conditioning") if isinstance(render_attempt.get("conditioning"), dict) else {}
    trace = conditioning.get("candidate_render_trace") if isinstance(conditioning.get("candidate_render_trace"), dict) else {}
    research_controls = _first_dict(
        trace.get("research_planner_render_controls") if isinstance(trace, dict) else None,
        conditioning.get("research_planner_render_controls") if isinstance(conditioning, dict) else None,
        conditioning.get("research_driven_blocker_solution_planner", {}).get("chosen_bundle") if isinstance(conditioning.get("research_driven_blocker_solution_planner"), dict) else None,
    )
    dml_controls = trace.get("v55_dml_lite_render_controls") if isinstance(trace.get("v55_dml_lite_render_controls"), dict) else {}
    direct = _first_dict(
        research_controls.get("v55_ivrcf_controlled_direct_activation") if isinstance(research_controls, dict) else None,
        research_controls.get("v53_ivrcf_initial_controls", {}).get("v55_controlled_direct_activation") if isinstance(research_controls.get("v53_ivrcf_initial_controls"), dict) else None,
        vmcr.get("v55_controlled_direct_activation") if isinstance(vmcr, dict) else None,
    )
    full_candidates = plan.get("full_render_candidates") if isinstance(plan.get("full_render_candidates"), list) else []
    first_full = full_candidates[0] if full_candidates and isinstance(full_candidates[0], dict) else {}
    static_moves = []
    dynamic_moves = []
    for src in (research_controls, dml_controls, direct, conditioning, first_full):
        if not isinstance(src, dict):
            continue
        if not static_moves and isinstance(src.get("static_moves"), list):
            static_moves = src.get("static_moves") or []
        if not dynamic_moves and isinstance(src.get("dynamic_moves"), list):
            dynamic_moves = src.get("dynamic_moves") or []
    if not static_moves and isinstance(direct.get("static_moves"), list):
        static_moves = direct.get("static_moves") or []
    if not dynamic_moves and isinstance(direct.get("dynamic_moves"), list):
        dynamic_moves = direct.get("dynamic_moves") or []
    applied_changes = direct.get("applied_changes") if isinstance(direct.get("applied_changes"), list) else []
    modules = direct.get("controlled_direct_modules") if isinstance(direct.get("controlled_direct_modules"), list) else []
    if not modules:
        modules = sorted({str(ch.get("module_id")) for ch in applied_changes if isinstance(ch, dict) and ch.get("module_id")})
    iteration_trace = vmcr.get("iteration_trace") if isinstance(vmcr.get("iteration_trace"), list) else []
    final_module_state = _first_dict(
        vmcr.get("final_module_state") if isinstance(vmcr.get("final_module_state"), dict) else None,
        vmcr.get("initial_module_state") if isinstance(vmcr.get("initial_module_state"), dict) else None,
        research_controls.get("ivrcf_initial_module_state") if isinstance(research_controls.get("ivrcf_initial_module_state"), dict) else None,
    )
    limiter_summary = {
        "limiter_release_ms": research_controls.get("limiter_release_ms", dml_controls.get("limiter_release_ms")),
        "limiter_lookahead_ms": research_controls.get("limiter_lookahead_ms", dml_controls.get("limiter_lookahead_ms")),
        "ceiling_delta_db": research_controls.get("ceiling_delta_db", dml_controls.get("ceiling_delta_db")),
        "density_recovery_limiter_release_ms": research_controls.get("density_recovery_limiter_release_ms", dml_controls.get("density_recovery_limiter_release_ms")),
        "density_recovery_limiter_lookahead_ms": research_controls.get("density_recovery_limiter_lookahead_ms", dml_controls.get("density_recovery_limiter_lookahead_ms")),
        "v55_density_lift_scale": research_controls.get("v55_density_lift_scale", dml_controls.get("v55_density_lift_scale")),
    }
    selected_label = cfr.get("selected") or vmcr.get("final_candidate_label") or first_full.get("candidate_label")
    export_accepted = fel.get("accepted")
    if export_accepted is None:
        export_accepted = not bool(cfr.get("final_export_lock_rejected_selected_candidate"))
    rollback_triggered = bool(cfr.get("rollback_triggered") or fel.get("rollback_to_standard_final") or cfr.get("fallback_final_source"))
    lineage_context = {
        "schema_version": "busy_final_listener_review_lineage_context_v8_5_3_55_3_3",
        "lineage_binding_policy": "Gemini review is attached to the exact delivered final-master lineage and is future-learning only; no current-render mutation.",
        "selected_candidate_label": selected_label,
        "selected_full_render_candidate_label": first_full.get("candidate_label"),
        "selected_final_source": cfr.get("fallback_final_source") or cfr.get("selected_final_source") or cfr.get("selected") or "selected_final_master",
        "final_export_lock_accepted": bool(export_accepted),
        "final_export_lock_rejected_selected_candidate": bool(cfr.get("final_export_lock_rejected_selected_candidate")),
        "final_export_lock_rejection_reasons": cfr.get("final_export_lock_rejection_reasons") or fel.get("mismatch_reasons") or [],
        "export_lock_mismatch_trace": {
            "schema_version": "busy_export_lock_mismatch_trace_v8_5_3_55_3_3",
            "accepted": bool(export_accepted),
            "rollback_to_standard_final": bool(fel.get("rollback_to_standard_final")),
            "mismatch_reasons": cfr.get("final_export_lock_rejection_reasons") or fel.get("mismatch_reasons") or [],
            "selected_candidate_label": selected_label,
            "selected_full_render_candidate_label": first_full.get("candidate_label"),
            "selected_final_source": cfr.get("fallback_final_source") or cfr.get("selected_final_source") or cfr.get("selected"),
            "fallback_final_source": cfr.get("fallback_final_source"),
            "policy": "Trace selected-candidate vs delivered-buffer identity when export lock rejects a candidate; learning uses this only as validation-priority context.",
        },
        "rollback_triggered": rollback_triggered,
        "fallback_final_source": cfr.get("fallback_final_source"),
        "commercial_result": {
            "target_lufs": cfr.get("target_lufs"),
            "commercial_floor_lufs": cfr.get("commercial_floor_lufs"),
            "final_lufs": (
                _analysis_summary_for_loudness(final_analysis if isinstance(final_analysis, dict) else {}).get("integrated_lufs")
                if isinstance(final_analysis, dict) else None
            ),
            "target_met": cfr.get("target_met"),
            "reason": cfr.get("reason"),
            "retry_blocked_by": cfr.get("retry_blocked_by"),
        },
        "v57_damage_aware_safe_selection": cfr.get("v57_damage_aware_safe_selection") if isinstance(cfr.get("v57_damage_aware_safe_selection"), dict) else {},
        "v57_commercial_loudness_feasibility": cfr.get("v57_commercial_loudness_feasibility") if isinstance(cfr.get("v57_commercial_loudness_feasibility"), dict) else {},
        "ivrcf": {
            "schema_version": vmcr.get("schema_version"),
            "active": vmcr.get("active"),
            "reason": vmcr.get("reason"),
            "final_candidate_label": vmcr.get("final_candidate_label"),
            "candidate_count_after": vmcr.get("candidate_count_after"),
            "iteration_count": len(iteration_trace),
            "iteration_stages": [str(x.get("stage")) for x in iteration_trace[:8] if isinstance(x, dict)],
            "initial_module_state_hash": _safe_hash_short(vmcr.get("initial_module_state")),
            "final_module_state_hash": _safe_hash_short(final_module_state),
        },
        "v55_controlled_direct": {
            "active": bool(direct.get("active", direct.get("direct_dsp_new_in_v55", False))),
            "direct_dsp_new_in_v55": bool(direct.get("direct_dsp_new_in_v55")),
            "controlled_direct_modules": _json_compact_clone(modules),
            "applied_change_count": len(applied_changes),
            "applied_change_modules": sorted({str(ch.get("module_id")) for ch in applied_changes if isinstance(ch, dict) and ch.get("module_id")}),
            "v55_1_prior_seen": bool(direct.get("v55_1_ivrcf_learning_prior_seen")),
        },
        "render_controls": {
            "static_move_count": len(static_moves),
            "dynamic_move_count": len(dynamic_moves),
            "static_moves_summary": [_compact_move(m) for m in static_moves[:12]],
            "dynamic_moves_summary": [_compact_move(m) for m in dynamic_moves[:16]],
            "low_side_mono_amount": research_controls.get("low_side_mono_amount", dml_controls.get("low_side_mono_amount")),
            "transient_micro_expander_amount": research_controls.get("transient_micro_expander_amount", dml_controls.get("transient_micro_expander_amount")),
            "limiter_control_summary": limiter_summary,
        },
        "decision_context": {
            "selected_profile": (decision if isinstance(decision, dict) else {}).get("selected_profile"),
            "selected_family": (decision if isinstance(decision, dict) else {}).get("selected_family"),
            "mastering_intensity": (decision if isinstance(decision, dict) else {}).get("mastering_intensity"),
        },
    }
    processing_summary = {
        "schema_version": "busy_final_listener_review_processing_summary_v8_5_3_55_3_3",
        "summary_hash": _safe_hash_short(lineage_context),
        "selected_candidate_label": selected_label,
        "final_source": lineage_context.get("selected_final_source"),
        "export_lock_accepted": lineage_context.get("final_export_lock_accepted"),
        "rollback_triggered": rollback_triggered,
        "v55_direct_active": lineage_context["v55_controlled_direct"].get("active"),
        "controlled_direct_modules": lineage_context["v55_controlled_direct"].get("controlled_direct_modules") or [],
        "ivrcf_iteration_count": lineage_context["ivrcf"].get("iteration_count"),
        "static_move_count": len(static_moves),
        "dynamic_move_count": len(dynamic_moves),
        "limiter_control_summary": limiter_summary,
        "v57_selected_damage_rejected_count": (cfr.get("v57_damage_aware_safe_selection") or {}).get("rejected_candidate_count") if isinstance(cfr.get("v57_damage_aware_safe_selection"), dict) else None,
        "v57_safe_push_budget_db": (cfr.get("v57_commercial_loudness_feasibility") or {}).get("safe_push_budget_db") if isinstance(cfr.get("v57_commercial_loudness_feasibility"), dict) else None,
        "strategy_context_status": ((decision if isinstance(decision, dict) else {}).get("historical_calibration_summary") or {}).get("strategy_context_status") if isinstance(((decision if isinstance(decision, dict) else {}).get("historical_calibration_summary") or {}), dict) else {},
        "learning_cohort_policy": ((decision if isinstance(decision, dict) else {}).get("historical_calibration_summary") or {}).get("learning_cohort_policy") if isinstance(((decision if isinstance(decision, dict) else {}).get("historical_calibration_summary") or {}), dict) else {},
    }
    return {
        "schema_version": "busy_final_listener_review_provenance_v8_5_3_55_3_3",
        "lineage_context": _json_compact_clone(lineage_context, max_list=32, max_dict=120, max_text=600),
        "processing_summary": _json_compact_clone(processing_summary, max_list=32, max_dict=120, max_text=600),
        "lineage_provenance_hash": processing_summary["summary_hash"],
        "learning_linkage": {
            "review_target": "delivered_final_master_highlight",
            "ivrcf_module_run_id": None,
            "export_qc_outcome_run_id": None,
            "links_by_run_id": True,
            "future_usage": "weak_listener_priority_hint_only_after_current_track_evidence",
        },
    }

def _run_gemini_final_master_highlight_review_best_effort(
    final: np.ndarray,
    sr: int,
    before: dict[str, Any],
    final_analysis: dict[str, Any],
    decision: dict[str, Any],
    commercial_finish_report: dict[str, Any],
    final_export_lock: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    """v8.5.3.55.3: Gemini listens only to the delivered final-master highlight.

    This post-export review never mutates the current render.  It stores a
    listener-facing technical review for Supabase learning/strategy context on
    future songs.
    """
    schema = "busy_gemini_final_master_highlight_review_v8_5_3_55_3_3"
    if not _env_bool("BUSY_GEMINI_FINAL_MASTER_HIGHLIGHT_REVIEW", True):
        return {"enabled": False, "available": False, "reason": "BUSY_GEMINI_FINAL_MASTER_HIGHLIGHT_REVIEW disabled", "schema_version": schema}
    try:
        duration_sec = float(len(final) / max(int(sr), 1)) if final is not None else 0.0
        target_base = float(os.environ.get("BUSY_GEMINI_FINAL_HIGHLIGHT_BASE_SEC", "34") or "34")
        target_min = float(os.environ.get("BUSY_GEMINI_FINAL_HIGHLIGHT_MIN_SEC", "30") or "30")
        target_max = float(os.environ.get("BUSY_GEMINI_FINAL_HIGHLIGHT_MAX_SEC", "40") or "40")
        adaptive_plan = build_adaptive_audio_preview_plan(
            duration_sec=duration_sec,
            features={
                **(before if isinstance(before, dict) else {}),
                "output_analysis": final_analysis if isinstance(final_analysis, dict) else {},
                "commercial_loudness_finish": commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
                "final_export_lock": final_export_lock if isinstance(final_export_lock, dict) else {},
            },
            segment_analysis=(before.get("segment_analysis", {}) if isinstance(before, dict) else {}),
            purpose="gemini_final_master_highlight_review",
            min_total_sec=target_min,
            base_total_sec=target_base,
            max_total_sec=target_max,
            default_segment_sec=target_base,
            max_segments=1,
        )
        # This is a single listener highlight, not a stitched raw genre preview.
        # The shared adaptive planner caps per-segment length for multi-window
        # previews, so overwrite the segment plan to one 30-40s focus window.
        focus_sec = float(adaptive_plan.get("target_total_sec") or target_base)
        if duration_sec > 0:
            focus_sec = min(max(8.0, focus_sec), duration_sec)
        adaptive_plan["segment_count"] = 1
        adaptive_plan["segment_sec"] = round(float(focus_sec), 3)
        adaptive_plan["focus_segment_sec"] = round(float(focus_sec), 3)
        adaptive_plan["policy"] = "adaptive single 30-40s final-master highlight for post-export listener/technical review; does not mutate current render"
        highlight_audio, highlight_source_summary = _build_preview_master_source_audio(
            final,
            int(sr),
            before if isinstance(before, dict) else {},
            total_sec=focus_sec,
            segment_sec=focus_sec,
            max_segments=1,
            adaptive_plan=adaptive_plan,
        )
        encoded = _encode_preview_master_wav_base64(
            highlight_audio,
            int(sr),
            target_sr=int(_env_float("BUSY_GEMINI_FINAL_MASTER_REVIEW_SR", 24000, 16000, 48000)),
        )
        highlight_summary = {k: v for k, v in encoded.items() if k != "base64"}
        highlight_summary.update(highlight_source_summary)
        highlight_summary["adaptive_preview_plan"] = adaptive_plan
        highlight_summary["source"] = "final_master_output_after_export_lock_and_qc"
        provenance = _build_final_listener_lineage_provenance(
            commercial_finish_report=commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
            final_export_lock=final_export_lock if isinstance(final_export_lock, dict) else {},
            final_analysis=final_analysis if isinstance(final_analysis, dict) else {},
            decision=decision if isinstance(decision, dict) else {},
        )
        highlight_summary["lineage_provenance_hash"] = provenance.get("lineage_provenance_hash")
        final_context = {
            "schema_version": "busy_final_master_highlight_review_context_v8_5_3_55_3_3",
            "output_analysis": _analysis_summary_for_loudness(final_analysis if isinstance(final_analysis, dict) else {}),
            "commercial_loudness_result": {
                "target_lufs": commercial_finish_report.get("target_lufs") if isinstance(commercial_finish_report, dict) else None,
                "commercial_floor_lufs": commercial_finish_report.get("commercial_floor_lufs") if isinstance(commercial_finish_report, dict) else None,
                "target_met": commercial_finish_report.get("target_met") if isinstance(commercial_finish_report, dict) else None,
                "reason": commercial_finish_report.get("reason") if isinstance(commercial_finish_report, dict) else None,
                "retry_blocked_by": commercial_finish_report.get("retry_blocked_by") if isinstance(commercial_finish_report, dict) else None,
            },
            "final_export_lock": {
                "accepted": final_export_lock.get("accepted") if isinstance(final_export_lock, dict) else None,
                "rollback_to_standard_final": final_export_lock.get("rollback_to_standard_final") if isinstance(final_export_lock, dict) else None,
                "mismatch_reasons": final_export_lock.get("mismatch_reasons") if isinstance(final_export_lock, dict) else [],
                "fallback_final_source": commercial_finish_report.get("fallback_final_source") if isinstance(commercial_finish_report, dict) else None,
            },
            "lineage_context": provenance.get("lineage_context") or {},
            "processing_summary": provenance.get("processing_summary") or {},
            "learning_linkage": provenance.get("learning_linkage") or {},
            "lineage_provenance_hash": provenance.get("lineage_provenance_hash"),
            "current_render_mutation_allowed": False,
            "future_learning_allowed": True,
        }
        from ai_decision.gemini_audio_judge import run_gemini_final_master_highlight_review
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("BUSY_GEMINI_API_KEY")
        review = run_gemini_final_master_highlight_review(
            api_key=gemini_key,
            preview_wav_base64=encoded.get("base64"),
            preview_summary=highlight_summary,
            final_context=final_context,
            features=before if isinstance(before, dict) else {},
            decision=decision if isinstance(decision, dict) else {},
            model=os.environ.get("BUSY_GEMINI_FINAL_MASTER_REVIEW_MODEL", os.environ.get("BUSY_GEMINI_AUDIO_MODEL", "gemini-2.5-flash-lite")) or "gemini-2.5-flash-lite",
            mode=mode,
        )
        review["enabled"] = True
        review["highlight_summary"] = highlight_summary
        review["final_context"] = final_context
        review["lineage_context"] = provenance.get("lineage_context") or {}
        review["processing_summary"] = provenance.get("processing_summary") or {}
        review["learning_linkage"] = provenance.get("learning_linkage") or {}
        review["lineage_provenance_hash"] = provenance.get("lineage_provenance_hash")
        review["policy"] = "post-export final master highlight review only; no current-render mutation; aggregate weak prior for future runs; lineage-bound to selected final render controls"
        return review
    except Exception as exc:
        return {"enabled": True, "available": False, "reason": "final_master_highlight_review_exception", "error": str(exc)[:700], "schema_version": schema}


def _preview_master_review_should_correct(review: dict[str, Any]) -> bool:
    if not isinstance(review, dict) or not review.get("available"):
        return False
    corrections = review.get("bounded_corrections") if isinstance(review.get("bounded_corrections"), dict) else {}
    verdict = str(review.get("verdict") or "").lower()
    if not bool(review.get("safe_to_full_render", True)):
        return True
    if verdict in {"needs_bounded_correction", "unsafe_to_full_render"}:
        return True
    try:
        if float(corrections.get("push_adjustment_db", 0.0) or 0.0) <= -0.05:
            return True
        if float(corrections.get("clip_drive_scale", 1.0) or 1.0) < 0.98:
            return True
    except Exception:
        pass
    return False


def _apply_preview_master_review_to_candidate(cand: dict[str, Any], review: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(cand) if isinstance(cand, dict) else {}
    corr = review.get("bounded_corrections") if isinstance(review.get("bounded_corrections"), dict) else {}
    try:
        push_adj = float(corr.get("push_adjustment_db", 0.0) or 0.0)
    except Exception:
        push_adj = 0.0
    if not bool(review.get("safe_to_full_render", True)) and push_adj > -0.05:
        push_adj = -0.35
    old_push = float(out.get("push_db", 0.0) or 0.0)
    out["push_db_before_gemini_preview_review"] = round(float(old_push), 3)
    out["push_db"] = round(float(max(0.18, old_push + max(-0.8, min(0.0, push_adj)))), 3)
    bundle = out.get("solution_bundle") if isinstance(out.get("solution_bundle"), dict) else {}
    bundle = copy.deepcopy(bundle)
    try:
        clip_scale = float(corr.get("clip_drive_scale", 1.0) or 1.0)
    except Exception:
        clip_scale = 1.0
    if clip_scale < 0.995:
        bundle["clip_drive_scale"] = min(float(bundle.get("clip_drive_scale", 1.0) or 1.0), max(0.65, min(1.0, clip_scale)))
        bundle["clip_mix_scale"] = min(float(bundle.get("clip_mix_scale", 1.0) or 1.0), max(0.70, min(1.0, clip_scale + 0.05)))
    dynamic_moves = [m for m in (bundle.get("dynamic_moves", []) or []) if isinstance(m, dict)]
    try:
        side_delta = float(corr.get("side_high_control_delta", 0.0) or 0.0)
    except Exception:
        side_delta = 0.0
    try:
        deess_delta = float(corr.get("deesser_delta", 0.0) or 0.0)
    except Exception:
        deess_delta = 0.0
    if side_delta > 0.01:
        dynamic_moves.append({"type": "bell", "freq": 9300, "gain_db": round(-2.0 * min(0.25, side_delta), 3), "q": 1.25, "target": "side", "source": "gemini_preview_master_bounded_side_high_control"})
    if deess_delta > 0.01:
        dynamic_moves.append({"type": "bell", "freq": 7200, "gain_db": round(-2.2 * min(0.25, deess_delta), 3), "q": 1.6, "target": "mid", "source": "gemini_preview_master_bounded_deesser"})
    if dynamic_moves:
        bundle["dynamic_moves"] = dynamic_moves
    try:
        width_delta = float(corr.get("width_delta", 0.0) or 0.0)
    except Exception:
        width_delta = 0.0
    if width_delta < -0.01:
        bundle["low_side_mono_amount"] = max(float(bundle.get("low_side_mono_amount", 0.0) or 0.0), min(0.35, abs(width_delta) * 2.0))
    out["solution_bundle"] = bundle
    out["gemini_preview_master_correction_applied"] = {
        "schema_version": "busy_gemini_preview_master_correction_v8_5_3_36",
        "source_verdict": review.get("verdict"),
        "safe_to_full_render": review.get("safe_to_full_render"),
        "push_adjustment_db": round(float(out["push_db"] - old_push), 3),
        "correction_hints": review.get("correction_hints", []),
        "policy": "bounded pre-full-render correction only; does not override deterministic QC or true-peak ceiling",
    }
    return out

def _commercial_loudness_finish_pass(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any],
    decision: dict[str, Any],
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Research-driven repair-then-push pass for Auto Commercial Master.

    v8.5.3.25 changes the hot path from "render a few hot candidates and reject
    bad ones" into a real blocker-resolution loop:

    render attempt -> measured blocker set -> research-driven solution planner ->
    revised DSP responsibility split -> re-render -> QC.  Only measured safe
    checkpoints can become final.  Failed attempts are kept in render_attempts,
    not in candidates.
    """
    profile = _commercial_loudness_finish_profile(decision, mode)
    if not profile.get("active"):
        return y, {**profile, "target_met": None, "retry_attempted": False}

    base = analyze_audio_fast_qc(y, sr, true_peak_oversample=_qc_oversample_factor())
    base_lufs = float(base.get("integrated_lufs", -99.0) or -99.0)
    target = float(profile.get("target_lufs", -8.9) or -8.9)
    floor = float(profile.get("commercial_floor_lufs", -9.7) or -9.7)
    target_met = bool(base_lufs >= floor)
    ownership = _v6351_mastering_ownership_decision(
        decision,
        base,
        {**profile, "target_lufs": target, "commercial_floor_lufs": floor, "target_met": target_met},
        before,
        stage_label="commercial_loudness_finish_base",
    )
    commercial_push_gate = _v6351_commercial_push_safety_gate_from_ownership(ownership)
    if isinstance(decision, dict):
        decision["mastering_ownership"] = ownership
        decision["commercial_push_safety_gate"] = commercial_push_gate
    report: dict[str, Any] = {
        **profile,
        "schema_version": "busy_commercial_loudness_finish_v8_5_3_63_5_1_1_mastering_ownership_metric_fallback",
        "input": _analysis_summary_for_loudness(base),
        "mastering_ownership": ownership,
        "commercial_push_safety_gate": commercial_push_gate,
        "selected_loudness_class": ownership.get("selected_loudness_class") if isinstance(ownership, dict) else None,
        "push_authority": ownership.get("push_authority") if isinstance(ownership, dict) else None,
        "target_met_before_retry": target_met,
        "target_met": target_met,
        "retry_attempted": False,
        "selected": "base",
        "render_attempts": [],
        "safe_checkpoints": [],
        # Backward-compatible alias: only QC-passed safe checkpoints are placed
        # here.  Failed render attempts are not candidates.
        "candidates": [],
        "candidate_wording_correction": "failed hot renders are render_attempts only; candidates/safe_checkpoints are QC-passed masters",
    }
    hp_v6314 = _v6314_handoff_policy_from_decision(decision)
    report["v6314_mastering_handoff_governance"] = {
        "schema_version": "busy_v6314_mastering_handoff_governance",
        "active": bool(hp_v6314.get("active")),
        "selected_mix_recipe": hp_v6314.get("selected_mix_recipe"),
        "punch_preserved": bool(hp_v6314.get("punch_preserved")),
        "prefer_cleaner_quieter_when_target_met": bool(hp_v6314.get("prefer_cleaner_quieter_when_target_met", True)) if hp_v6314 else False,
        "max_crest_loss_db": hp_v6314.get("max_crest_loss_db"),
        "policy": "v63.1.4: soften final chase around punch-preserved crest loss; repeated crest_loss_relative shrinks future push cap",
    }

    # v8.5.3.35: enforce the optimized render-graph budget inside the
    # commercial finish pass itself.  Candidate generation may still create a
    # richer parameter-space plan, but only the preview-ranked finalist(s) below
    # are allowed to become full-track render attempts.
    pipeline_budget = render_budget_policy() if callable(render_budget_policy) else {}
    try:
        _budget_normal_full_renders = int(pipeline_budget.get("normal_full_renders", 1) or 1)
    except Exception:
        _budget_normal_full_renders = 1
    try:
        _budget_hard_full_render_ceiling = int(pipeline_budget.get("hard_full_render_ceiling", 2) or 2)
    except Exception:
        _budget_hard_full_render_ceiling = 2
    intensity_for_budget = str(profile.get("mastering_intensity") or _mastering_intensity_key() or "balanced").strip().lower()
    v47_diversity_enabled = _env_bool("BUSY_V47_LOUDNESS_CHECKPOINT_DIVERSITY", True)
    v47_diversity_budget = int(_env_float("BUSY_V47_LOUDNESS_DIVERSITY_FULL_RENDERS", 3, 2, 4)) if v47_diversity_enabled else 2
    _effective_full_render_ceiling = max(int(_budget_hard_full_render_ceiling), int(v47_diversity_budget)) if v47_diversity_enabled else int(_budget_hard_full_render_ceiling)
    env_full_render_budget = int(_env_float(
        "BUSY_COMMERCIAL_FINISH_MAX_FULL_RENDERS",
        float(min(_effective_full_render_ceiling, v47_diversity_budget)),
        1,
        max(1, min(4, _effective_full_render_ceiling)),
    ))
    commercial_full_render_budget = max(1, min(int(env_full_render_budget), max(1, min(4, int(_effective_full_render_ceiling)))))
    if _env_bool("BUSY_V48_LRA_CANDIDATE_GENERATOR", True) and intensity_for_budget != "safe":
        commercial_full_render_budget = max(commercial_full_render_budget, int(_env_float("BUSY_V48_MIN_FULL_RENDERS", 4, 3, 4)))
    if intensity_for_budget == "safe":
        commercial_full_render_budget = 1
    v48_2_2_virtual_proxy_enabled = _env_bool("BUSY_V48_2_2_VIRTUAL_PROXY_RENDER", True)
    if v48_2_2_virtual_proxy_enabled and _env_bool("BUSY_V48_2_2_SELECTED_ONLY_FULL_RENDER", True):
        commercial_full_render_budget = 1
    report["optimized_render_budget_enforcement"] = {
        "schema_version": "busy_commercial_finish_render_budget_v8_5_3_35",
        "active": True,
        "normal_full_renders": int(_budget_normal_full_renders),
        "hard_full_render_ceiling": int(_budget_hard_full_render_ceiling),
        "v47_diversity_enabled": bool(v47_diversity_enabled),
        "v47_diversity_full_render_budget": int(v47_diversity_budget),
        "effective_full_render_ceiling": int(_effective_full_render_ceiling),
        "commercial_finish_full_render_budget": int(commercial_full_render_budget),
        "intermediate_wav_renders_allowed": 0,
        "policy": "candidate plans may be generated in parameter space; v47.1 can full-render up to three preview-ranked finalists for loudness checkpoint diversity while deterministic QC remains mandatory",
    }

    virtual_hot_solver = build_virtual_hot_commercial_solver(y, sr, before, decision, mode, base, profile)
    report["virtual_hot_commercial_solver"] = virtual_hot_solver
    report["representative_section_map"] = virtual_hot_solver.get("representative_section_map", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["blocker_capacity_map"] = virtual_hot_solver.get("blocker_capacity_map", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["blocker_solution_plan"] = virtual_hot_solver.get("blocker_solution_plan", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["final_push_strategy"] = virtual_hot_solver.get("final_push_strategy", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["musical_role_identity_map"] = virtual_hot_solver.get("musical_role_identity_map", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["virtual_premaster_strategy"] = {
        "schema_version": "busy_virtual_premaster_strategy_v8_5_3_27",
        "basis": "primary evidence / GPT-mini text opinion / Gemini-Realtime audio opinions / pressure-test blocker map / measured retry trace",
        "premaster_moves": report.get("blocker_solution_plan", {}),
        "limiting_strategy": report.get("final_push_strategy", {}),
        "policy": "initial strategy is not final; every failure is re-planned from measured blocker trace until a safe checkpoint or convergence stop",
    }
    report["multi_agent_virtual_premaster_strategy_council"] = _build_multi_agent_virtual_premaster_strategy_council(before, base, decision, virtual_hot_solver, profile)
    report["problem_resolution_loop"] = {
        "schema_version": "busy_problem_resolution_loop_v8_5_3_27",
        "active": True,
        "fixed_retry_limit": False,
        "loop_policy": "retry count is not musical policy; stop only on safe checkpoint, no blocker, non-useful push, or measured convergence without damage reduction",
        "planner": "research_driven_blocker_solution_planner",
        "failed_attempts_go_to": "render_attempts",
        "qc_passed_attempts_go_to": "safe_checkpoints/candidates",
    }

    force_finish_eval_when_floor_met = bool(_env_bool("BUSY_V6316_FORCE_COMMERCIAL_FINISH_EVALUATION", True))
    report["v6316_commercial_finish_reconnect"] = {
        "schema_version": "busy_v6316_commercial_finish_reconnect",
        "active": bool(force_finish_eval_when_floor_met),
        "base_target_met_before_retry": bool(target_met),
        "policy": "Do not short-circuit the commercial finish solely because the input checkpoint already meets the floor; at least evaluate the existing commercial candidate/finalizer path unless explicitly disabled.",
    }
    if target_met and not _env_bool("BUSY_VIRTUAL_HOT_SOLVER_PUSH_WHEN_FLOOR_MET", False) and not force_finish_eval_when_floor_met:
        report["reason"] = "commercial_floor_already_met"
        report["output"] = report["input"]
        return y, report

    gap_to_floor = max(0.0, floor - base_lufs)
    gap_to_target = max(gap_to_floor, target - base_lufs)
    max_push = float(profile.get("max_push_db", 2.0) or 2.0)
    hp_v6314 = _v6314_handoff_policy_from_decision(decision)
    if hp_v6314.get("active"):
        _before_cap = float(max_push)
        if hp_v6314.get("punch_preserved"):
            max_push = min(max_push, _env_float("BUSY_BAMIX_V6314_PUNCH_FINISH_PUSH_CAP_DB", float(hp_v6314.get("max_finish_push_db", max_push) or max_push), 0.2, 3.0))
        try:
            max_push = min(max_push, float(hp_v6314.get("max_finish_push_db", max_push) or max_push))
        except Exception:
            pass
        report["v6314_mastering_handoff_governance"]["max_push_before_handoff_cap_db"] = round(float(_before_cap), 3)
        report["v6314_mastering_handoff_governance"]["max_push_after_handoff_cap_db"] = round(float(max_push), 3)
    strategy = virtual_hot_solver.get("final_push_strategy", {}) if isinstance(virtual_hot_solver, dict) and isinstance(virtual_hot_solver.get("final_push_strategy", {}), dict) else {}
    solver_push = float(strategy.get("recommended_input_gain_db", 0.0) or 0.0) if strategy else 0.0
    if solver_push > 0.0:
        # v8.5.3.34: the final intensity lock is a hard render-budget boundary.
        # A virtual solver may suggest a theoretical push, but it cannot expand
        # Safe/Balanced max_push outside the selected bucket.
        if str(profile.get("mastering_intensity") or "").lower() in {"safe", "balanced"}:
            max_push = float(profile.get("max_push_db", max_push) or max_push)
        else:
            max_push = max(max_push, float(solver_push) + 0.35, gap_to_target + 0.25)
    if isinstance(ownership, dict) and ownership.get("active"):
        _before_ownership_cap = float(max_push)
        _ownership_cap = _v6351_first_num(ownership.get("allowed_final_push_db"), ownership.get("final_push_budget_db"), default=max_push)
        max_push = min(float(max_push), max(0.0, float(_ownership_cap or 0.0)))
        profile["max_push_db_before_v6351_ownership"] = round(float(_before_ownership_cap), 3)
        profile["max_push_db"] = round(float(max_push), 3)
        report["v6351_ownership_push_cap"] = {
            "schema_version": "busy_v6351_ownership_push_cap",
            "active": True,
            "max_push_before_db": round(float(_before_ownership_cap), 3),
            "max_push_after_db": round(float(max_push), 3),
            "push_authority": ownership.get("push_authority"),
            "selected_loudness_class": ownership.get("selected_loudness_class"),
            "target_miss_allowed": bool(ownership.get("artifact_protected_fallback")),
            "policy": "Commercial finish candidate input gain is capped by ownership before v57/v60/v61 QC; target miss is allowed in artifact-protected fallback.",
        }
        if ownership.get("artifact_protected_fallback") and not target_met:
            report["reason"] = "v6351_artifact_protected_fallback_before_retry"
            report["target_miss_allowed_by_ownership"] = True
            report["output"] = report["input"]
            return y, report
    needed = float(np.clip(max(gap_to_target, solver_push), 0.0, max_push))
    if needed < 0.20 and not (target_met and force_finish_eval_when_floor_met):
        report["reason"] = "gap_too_small" if not target_met else "commercial_floor_already_met"
        report["output"] = report["input"]
        return y, report
    if needed < 0.20 and target_met and force_finish_eval_when_floor_met:
        report["v6316_commercial_finish_reconnect"]["small_gap_promoted_to_minimum_candidate_eval"] = True
        needed = min(max_push, 0.20) if max_push >= 0.20 else max_push

    solver_pushes = strategy.get("recommended_candidate_gains_db", []) if isinstance(strategy.get("recommended_candidate_gains_db", []), list) else []
    pushes = [float(x) for x in solver_pushes if isinstance(x, (int, float)) and float(x) >= 0.18]
    if not pushes:
        pushes = [min(needed, x) for x in (0.55, 0.95, 1.35, 1.80, 2.35, max_push)]
    elif needed not in pushes:
        pushes.append(needed)
    pushes = sorted({round(float(np.clip(x, 0.18, max_push)), 3) for x in pushes if x >= 0.18})
    if isinstance(report.get("v6314_mastering_handoff_governance"), dict):
        report["v6314_mastering_handoff_governance"]["initial_push_schedule_db"] = list(pushes)
    max_finish_starts = int(_env_float("BUSY_COMMERCIAL_FINISH_MAX_START_PUSHES", 8 if profile.get("adaptive_hot_commercial") or profile.get("experimental_hot_commercial") else 5, 1, 14))
    if len(pushes) > max_finish_starts:
        focus = solver_push if solver_push > 0 else needed
        pushes = sorted(pushes, key=lambda x: (abs(x - focus), x))[: max_finish_starts - 1] + [max(pushes)]
        pushes = sorted(set(pushes))

    base_warnings = _safety_risk(before, base, decision)
    v57_feasibility = _v57_safe_push_feasibility(before, base, decision, profile, floor, target, base_warnings)
    report["v57_commercial_loudness_feasibility"] = v57_feasibility
    if v57_feasibility.get("enabled") and not v57_feasibility.get("useful_safe_push_available"):
        report["push_schedule_before_v57"] = pushes
        pushes = []
        max_push = 0.0
        profile["max_push_db"] = 0.0
        report["push_schedule_after_v57"] = pushes
        report["reason"] = "v57_no_safe_push_budget_select_base_or_existing_safe_candidate"
        report["retry_blocked_by"] = v57_feasibility.get("reasons")
    elif v57_feasibility.get("enabled"):
        budget = float(v57_feasibility.get("safe_push_budget_db", 0.0) or 0.0)
        report["push_schedule_before_v57"] = pushes
        max_push = min(float(max_push), float(budget))
        profile["max_push_db_before_v57"] = profile.get("max_push_db")
        profile["max_push_db"] = float(max_push)
        pushes = sorted({round(float(min(x, budget)), 3) for x in pushes if float(min(x, budget)) >= 0.18})
        if not pushes and budget >= 0.18:
            pushes = [round(float(budget), 3)]
        report["push_schedule_after_v57"] = pushes
    base_hard = _commercial_loudness_candidate_hard_blocks(before, base, decision, profile, base_warnings)
    best_audio = y
    best_analysis = base
    best_score = -1e9
    best_name = "base"
    base_problem_report = _candidate_problem_set(before, base, decision, profile, base_warnings, base_problem_set=[])
    report["retry_attempted"] = True
    report["base_warnings"] = base_warnings
    report["base_hard_blocks"] = base_hard
    report["base_problem_set"] = base_problem_report
    report["candidate_selection_policy"] = "only QC-passed safe_checkpoints can be selected; failed hot renders are not candidates"

    # v48.2 MDT: calibrate candidate QC thresholds from audio evidence and
    # structural microdynamic profile.  This is not a genre-name exception and
    # it never changes EQ/compression/limiter/gain parameters.  It only adjusts
    # candidate accept/reject and near-miss eligibility thresholds when measured
    # audio evidence supports the calibration.
    try:
        micro_tol = build_microdynamic_tolerance_report(
            base_analysis=base,
            candidate_analysis=None,
            profile=profile,
            decision=decision,
            context={"stage": "commercial_loudness_finish_profile_calibration"},
        )
        old_hard_crest = float(profile.get("hard_crest_min_db", profile.get("crest_hard_min_db", 6.5)) or 6.5)
        new_hard_crest = float(((micro_tol.get("thresholds_blended", {}) or {}).get("crest_hard_floor_db", old_hard_crest)) or old_hard_crest)
        if micro_tol.get("applied_to_candidate_qc") and new_hard_crest < old_hard_crest:
            profile["pre_microdynamic_hard_crest_min_db"] = round(float(old_hard_crest), 3)
            profile["hard_crest_min_db"] = round(float(new_hard_crest), 3)
            new_warn = float(((micro_tol.get("thresholds_blended", {}) or {}).get("crest_warning_floor_db", profile.get("crest_min_db", new_hard_crest + 0.5))) or (new_hard_crest + 0.5))
            profile["crest_min_db"] = min(float(profile.get("crest_min_db", new_warn) or new_warn), float(new_warn))
        profile["microdynamic_tolerance"] = micro_tol
        report["microdynamic_tolerance"] = micro_tol
        report["hard_crest_min_db"] = profile.get("hard_crest_min_db")
        report["crest_min_db"] = profile.get("crest_min_db")
    except Exception as exc:
        report["microdynamic_tolerance"] = {
            "schema_version": "busy_microdynamic_tolerance_v8_5_3_48_2",
            "enabled": bool(_env_bool("BUSY_V48_2_MICRODYNAMIC_TOLERANCE", True)),
            "active": False,
            "reason": "calibration_exception",
            "error": str(exc)[:240],
            "policy": "fallback to existing conservative profile thresholds",
        }

    v48_pde_enabled = bool(_env_bool("BUSY_V48_PDE", True))
    v48_nearmiss_enabled = bool(_env_bool("BUSY_V48_NEARMISS_REPAIR", True))
    v48_lra_generator_enabled = bool(_env_bool("BUSY_V48_LRA_CANDIDATE_GENERATOR", True))
    v48_section_planner_enabled = bool(_env_bool("BUSY_V48_SECTION_AWARE_PLANNER", True))
    v48_2_ssrw_enabled = bool(_env_bool("BUSY_V48_2_SSRW", True))
    ssrw_source = {}
    if isinstance(before, dict) and isinstance(before.get("stem_sum_residue_witness"), dict):
        ssrw_source = before.get("stem_sum_residue_witness") or {}
    elif isinstance(before, dict) and isinstance(before.get("multi_component_assist"), dict):
        mca0 = before.get("multi_component_assist") or {}
        if isinstance(mca0.get("stem_sum_residue_witness"), dict):
            ssrw_source = mca0.get("stem_sum_residue_witness") or {}
    if not ssrw_source and isinstance(decision, dict):
        ctx_ssrw = ((decision.get("final_decision_context") or {}).get("stem_sum_residue_witness") if isinstance(decision.get("final_decision_context"), dict) else None)
        if isinstance(ctx_ssrw, dict):
            ssrw_source = ctx_ssrw
    ssrw_summary = compact_ssrw_summary(ssrw_source) if v48_2_ssrw_enabled else {"enabled": False, "reason": "disabled"}
    report["stem_sum_residue_witness"] = ssrw_summary
    report["v48_active_candidate_strategy"] = {
        "schema_version": "busy_v48_active_candidate_strategy_v8_5_3_48_2_1_render_blocker_ai_convergence",
        "pde_enabled": v48_pde_enabled,
        "nearmiss_repair_enabled": v48_nearmiss_enabled,
        "lra_candidate_generator_enabled": v48_lra_generator_enabled,
        "section_aware_planner_enabled": v48_section_planner_enabled,
        "v48_1_section_planner_enabled": bool(_env_bool("BUSY_V48_1_SECTION_PLANNER", True)),
        "v48_1_section_gain_envelope_enabled": bool(_env_bool("BUSY_V48_1_SECTION_GAIN_ENVELOPE", True)),
        "v48_2_ssrw_enabled": v48_2_ssrw_enabled,
        "v48_2_ssrw_available": bool(ssrw_summary.get("available")),
        "v48_2_ssrw_reliability": ssrw_summary.get("global_reliability"),
        "v48_2_microdynamic_tolerance_enabled": bool(_env_bool("BUSY_V48_2_MICRODYNAMIC_TOLERANCE", True)),
        "v48_2_microdynamic_profile": (report.get("microdynamic_tolerance") or {}).get("global_profile") if isinstance(report.get("microdynamic_tolerance"), dict) else None,
        "v48_2_microdynamic_confidence": (report.get("microdynamic_tolerance") or {}).get("profile_confidence") if isinstance(report.get("microdynamic_tolerance"), dict) else None,
        "candidate_count_policy": "v48.2.1 uses convergence-based early stop instead of treating render count as a musical rule; static render counts remain worker-safety caps only",
        "fallback_policy": "if PDE/MDT/SSRW and legacy-blocker reconciliation cannot create a deterministic safe checkpoint, base/safe-checkpoint fallback remains authoritative",
        "ai_resource_allocation_policy": {
            "schema_version": "busy_ai_resource_allocation_v8_5_3_48_2_1",
            "active": True,
            "realtime_role": "candidate_audio_comparator_only_when_top_candidates_are_close",
            "gemini_role": "final_quality_validator_for_nearmiss_or_risk_flags",
            "gpt54mini_role": "stage_summary_public_safe_explanation_and_gpt55_packet_builder",
            "gpt55_role": "hard_case_arbitration_within_allowed_candidates_only",
            "hard_qc_is_law": True,
            "no_ai_dsp_control": True,
            "static_call_counts_are_safety_caps_not_musical_policy": True
        },
    }

    def _pde_context_for_candidate(extra: dict[str, Any] | None = None) -> dict[str, Any]:
        lgcca = decision.get("lgcca_influence_routing", {}) if isinstance(decision, dict) and isinstance(decision.get("lgcca_influence_routing", {}), dict) else {}
        lr = lgcca.get("limiter_risk_budget", {}) if isinstance(lgcca.get("limiter_risk_budget", {}), dict) else {}
        risk = 0.0
        for k in ("vocal_dullness_risk", "transient_loss_risk", "low_end_pumping_risk"):
            try:
                risk = max(risk, float(lr.get(k, 0.0) or 0.0))
            except Exception:
                pass
        ctx = {"lgcca_limiter_risk": float(np.clip(risk, 0.0, 1.0))}
        if isinstance(extra, dict):
            ctx.update(extra)
        return ctx

    def _attach_pde_to_attempt(rep: dict[str, Any], cand_analysis: dict[str, Any]) -> dict[str, Any]:
        if not v48_pde_enabled or not isinstance(rep, dict) or not isinstance(cand_analysis, dict):
            return rep
        blockers = list(rep.get("unresolved_blocker_set", []) or []) + list(rep.get("hard_blocks", []) or [])
        ssrw_ctx = build_ssrw_pde_context(ssrw_source, base, cand_analysis) if v48_2_ssrw_enabled else {}
        pde = build_pde_score(base_analysis=base, candidate_analysis=cand_analysis, profile=profile, context=_pde_context_for_candidate(ssrw_ctx))
        nm = build_near_miss_eligibility(
            base_analysis=base,
            candidate_analysis=cand_analysis,
            blockers=blockers,
            pde=pde,
            min_lufs_advantage=_env_float("BUSY_V48_NEARMISS_MIN_LUFS_ADVANTAGE", 0.6, 0.1, 2.0),
            profile=profile,
            max_crest_deficit_db=_env_float("BUSY_V48_NEARMISS_MAX_CREST_DEFICIT_DB", 0.5, 0.0, 2.0),
        )
        rep["pde"] = pde
        rep["nearmiss_eligibility"] = nm
        _reconcile_legacy_commercial_blockers(rep, cand_analysis)
        return rep

    def _reconcile_legacy_commercial_blockers(rep: dict[str, Any], cand_analysis: dict[str, Any]) -> dict[str, Any]:
        """Demote superseded legacy commercial blockers when new v48 evidence is clean.

        Older commercial-finish logic could reject a candidate with
        damage_score_exceeds_loudness_benefit even after PDE/MDT/SSRW judged the
        candidate clean.  This function does not create a genre exception and it
        never overrides hard vetoes.  It only reclassifies legacy soft blockers
        when the candidate already passes modern deterministic evidence.
        """
        if not _env_bool("BUSY_V48_2_1_LEGACY_BLOCKER_RECONCILIATION", True):
            rep["legacy_blocker_reconciliation"] = {
                "schema_version": "busy_legacy_commercial_blocker_reconciliation_v8_5_3_48_2_1",
                "active": False,
                "reason": "disabled_by_env",
            }
            return rep
        pde = rep.get("pde") if isinstance(rep.get("pde"), dict) else {}
        decision_zone = str(pde.get("decision_zone") or "")
        pde_hard = [str(x) for x in (pde.get("hard_vetoes", []) or [])]
        unresolved = [str(x) for x in (rep.get("unresolved_blocker_set", []) or []) if str(x)]
        hard_blocks = [str(x) for x in (rep.get("hard_blocks", []) or []) if str(x)]
        strict_hard_names = {
            "true_peak_too_hot", "tp_hard", "stereo_hard",
            "stereo_correlation_too_low", "mono_collapse_risk",
            "commercial_correlation_floor", "commercial_low_side_floor",
            "vocal_hard", "vocal_mask_violation", "render_audit_hard",
            "lgcca_evidence_hard", "ssrw_no_injection_audit_hard",
            "playback_translation_hard", "pump_hard",
        }
        legacy_soft = {"damage_score_exceeds_loudness_benefit"}
        conditional_soft = {"lra_too_flat", "lra_reduced_too_much"}
        soft_candidates = set(unresolved) & (legacy_soft | conditional_soft)
        non_soft_unresolved = [x for x in unresolved if x not in (legacy_soft | conditional_soft)]
        terms = pde.get("terms", {}) if isinstance(pde.get("terms"), dict) else {}
        try:
            ssrw_residue = float(terms.get("d_ssrw_residue_gain", 0.0) or 0.0)
        except Exception:
            ssrw_residue = 0.0
        try:
            ssrw_body = float(terms.get("d_ssrw_body_loss", 0.0) or 0.0)
        except Exception:
            ssrw_body = 0.0
        try:
            tp = float(cand_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0)
            lufs = float(cand_analysis.get("integrated_lufs", -99.0) or -99.0)
        except Exception:
            tp, lufs = 99.0, -99.0
        lufs_gain = float(lufs - base_lufs)
        strict_hard_present = sorted(set(hard_blocks + pde_hard) & strict_hard_names)
        modern_accept = bool(decision_zone in {"strong_accept_zone", "weak_accept_zone"} and not pde_hard)
        tp_safe = bool(tp <= float(FINAL_TRUE_PEAK_CEILING_DB) + _env_float("BUSY_V48_2_1_RECONCILE_TP_MARGIN_DB", 0.03, 0.0, 0.12))
        ssrw_clean = bool(ssrw_residue < _env_float("BUSY_V48_2_1_SSRW_RESIDUE_SOFT_LIMIT", 0.35, 0.0, 1.0) and ssrw_body < _env_float("BUSY_V48_2_1_SSRW_BODY_SOFT_LIMIT", 0.55, 0.0, 1.0))
        loudness_useful = bool(lufs_gain > _env_float("BUSY_V48_2_1_RECONCILE_MIN_LUFS_GAIN", 0.18, 0.0, 1.0))
        can_supersede = bool(
            soft_candidates
            and not non_soft_unresolved
            and modern_accept
            and not strict_hard_present
            and tp_safe
            and ssrw_clean
            and loudness_useful
        )
        rec = {
            "schema_version": "busy_legacy_commercial_blocker_reconciliation_v8_5_3_48_2_1",
            "active": True,
            "modern_decision_zone": decision_zone,
            "legacy_soft_blockers": sorted(soft_candidates),
            "non_soft_unresolved_blockers": sorted(non_soft_unresolved),
            "strict_hard_present": strict_hard_present,
            "tp_safe": bool(tp_safe),
            "ssrw_clean": bool(ssrw_clean),
            "lufs_gain_vs_base_db": round(float(lufs_gain), 3),
            "superseded": bool(can_supersede),
            "policy": "legacy commercial blockers become soft notes only when PDE/MDT/SSRW and hard QC are clean; hard vetoes remain non-overridable.",
        }
        if can_supersede:
            new_unresolved = [x for x in unresolved if x not in soft_candidates]
            rep["unresolved_blocker_set"] = sorted(set(new_unresolved))
            rep["hard_blocks"] = [x for x in hard_blocks if x not in soft_candidates]
            acc = rep.get("acceptance") if isinstance(rep.get("acceptance"), dict) else {}
            acc["accepted"] = True
            acc["unresolved_blocker_set"] = sorted(set(new_unresolved))
            acc["legacy_blockers_superseded_by_v48_2_1"] = sorted(soft_candidates)
            acc["decision_authority"] = "pde_mdt_ssrw_legacy_reconciliation"
            rep["acceptance"] = acc
            rep["qc_passed_safe_checkpoint"] = True
            rep["candidate_semantics"] = "safe_checkpoint_eligible_after_v48_2_1_legacy_blocker_reconciliation"
        else:
            rec["reason"] = "not_superseded_due_to_non_soft_or_hard_risk_or_insufficient_modern_acceptance"
        rep["legacy_blocker_reconciliation"] = rec
        return rep

    def _repair_nearmiss_candidate(
        cand_audio: np.ndarray,
        cand_analysis: dict[str, Any],
        rep: dict[str, Any],
        *,
        label: str,
        push_db: float,
    ) -> tuple[np.ndarray | None, dict[str, Any] | None, dict[str, Any] | None]:
        if not v48_nearmiss_enabled or not isinstance(rep, dict) or not isinstance(cand_analysis, dict):
            return None, None, None
        nm = rep.get("nearmiss_eligibility") if isinstance(rep.get("nearmiss_eligibility"), dict) else {}
        if not nm.get("eligible"):
            return None, None, {"enabled": v48_nearmiss_enabled, "eligible": False, "reason": "not_nearmiss_eligible", "eligibility": nm}
        actions: list[str] = []
        repaired = np.asarray(cand_audio, dtype=np.float32)
        pre_tp = float(cand_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0)
        target_ceiling = float(FINAL_TRUE_PEAK_CEILING_DB)
        # Single-pass repair: small pull/TP normalize/crest-preserving relax.
        pull_db = 0.0
        if pre_tp > target_ceiling:
            pull_db = max(-0.5, min(-0.2, target_ceiling - pre_tp - 0.03))
            repaired = apply_gain_db(repaired, pull_db)
            actions.append("pre_limiter_pull")
        blockers = set((rep.get("unresolved_blocker_set") or []) + (rep.get("hard_blocks") or []))
        if any(x in blockers for x in ["crest_factor_too_low", "commercial_crest_hard_floor", "damage_score_exceeds_loudness_benefit"]):
            repaired = transient_micro_expander(repaired, sr, amount=_env_float("BUSY_V48_NEARMISS_TRANSIENT_AMOUNT", 0.025, 0.0, 0.08))
            actions.append("transient_micro_expander")
        post_limiter_tp = None
        if pre_tp > target_ceiling or "true_peak_too_hot" in blockers:
            repaired = lookahead_limiter(repaired, sr, ceiling_db=target_ceiling, lookahead_ms=1.0, release_ms=160)
            # Near-miss TP repair must never use positive makeup gain.  v48.0
            # could normalize upward after the safety limiter, making a repaired
            # candidate slightly louder and hotter than its parent.
            post_limiter_tp = float(true_peak_db_oversampled_chunked(repaired, oversample=_qc_oversample_factor(), sr=sr))
            tp_gain = min(0.0, float(target_ceiling) - post_limiter_tp)
            if tp_gain < -1e-6:
                repaired = apply_gain_db(repaired, tp_gain)
            actions.append("micro_safety_limiter_true_peak_no_makeup_normalize")
        else:
            tp_gain = 0.0
        repair_analysis = analyze_audio_fast_qc(repaired, sr, true_peak_oversample=_qc_oversample_factor())
        repair_warnings = _safety_risk(before, repair_analysis, decision)
        repair_hard = _commercial_loudness_candidate_hard_blocks(before, repair_analysis, decision, profile, repair_warnings)
        repair_acceptance = _candidate_acceptance_decision(
            before,
            repair_analysis,
            decision,
            profile,
            repair_warnings + repair_hard,
            base,
            list(base_problem_report.get("problem_set", []) or []),
            floor,
            target,
        )
        repair_ssrw_ctx = build_ssrw_pde_context(ssrw_source, base, repair_analysis) if v48_2_ssrw_enabled else {}
        repair_ssrw_ctx.update({"tp_repair_events": 1.0 if actions else 0.0})
        repair_pde = build_pde_score(base_analysis=base, candidate_analysis=repair_analysis, profile=profile, context=_pde_context_for_candidate(repair_ssrw_ctx))
        post_lufs = float(repair_analysis.get("integrated_lufs", -99.0) or -99.0)
        post_tp = float(repair_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0)
        post_adv = post_lufs - base_lufs
        post_tp_safe = bool(post_tp <= target_ceiling + 1e-3)
        accepted = bool(
            repair_acceptance.get("accepted")
            and not repair_hard
            and post_tp_safe
            and post_adv >= _env_float("BUSY_V48_NEARMISS_POST_MIN_LUFS_ADVANTAGE", 0.4, 0.0, 2.0)
            and float(repair_pde.get("ratio", 999.0) or 999.0) <= _env_float("BUSY_V48_NEARMISS_MAX_PDE_RATIO", 1.0, 0.5, 1.5)
            and not repair_pde.get("hard_vetoes")
        )
        repair_rep = {
            "name": f"nearmiss_repair_{label}_{push_db:.2f}db",
            "candidate_label": f"{label}_nearmiss_repaired",
            "attempt_type": "nearmiss_repair",
            "parent_attempt": rep.get("name"),
            "single_pass_only": True,
            "repair_actions": actions,
            "pre_metrics": rep.get("analysis", _analysis_summary_for_loudness(cand_analysis)),
            "post_metrics": _analysis_summary_for_loudness(repair_analysis),
            "analysis": _analysis_summary_for_loudness(repair_analysis),
            "pre_limiter_pull_db": round(float(pull_db), 3),
            "true_peak_normalize_gain_db": round(float(tp_gain), 3),
            "post_limiter_true_peak_before_no_makeup_db": round(float(post_limiter_tp), 3) if post_limiter_tp is not None else None,
            "no_positive_tp_makeup": True,
            "post_repair_true_peak_safe": bool(post_tp_safe),
            "warnings": repair_warnings,
            "hard_blocks": repair_hard,
            "acceptance": repair_acceptance,
            "problem_set": repair_acceptance.get("problem_set", {}),
            "unresolved_blocker_set": sorted(set(repair_acceptance.get("unresolved_blocker_set", []) or repair_hard or [])),
            "pde": repair_pde,
            "qc_passed_safe_checkpoint": bool(accepted),
            "accepted": bool(accepted),
            "reject_reasons": [] if accepted else sorted(set((repair_acceptance.get("unresolved_blocker_set", []) or []) + repair_hard + list(repair_pde.get("hard_vetoes", []) or []) + (["pde_ratio_over_limit"] if float(repair_pde.get("ratio", 999.0) or 999.0) > _env_float("BUSY_V48_NEARMISS_MAX_PDE_RATIO", 1.0, 0.5, 1.5) else []) + (["post_lufs_advantage_below_0_4"] if post_adv < _env_float("BUSY_V48_NEARMISS_POST_MIN_LUFS_ADVANTAGE", 0.4, 0.0, 2.0) else []) + (["post_true_peak_still_hot"] if not post_tp_safe else []))),
            "ssrw_nearmiss_modifier": repair_ssrw_ctx if isinstance(repair_ssrw_ctx, dict) else {},
            "policy": "v48 near-miss repair is single-pass, repairable-blocker-only, fully reanalyzed, and v48.2 SSRW can only demote residue-like risk.",
        }
        return (np.nan_to_num(repaired, nan=0.0, posinf=0.0, neginf=0.0), repair_analysis, repair_rep)

    def _v48_extra_candidate_plans(existing: list[dict[str, Any]], max_push_db: float) -> list[dict[str, Any]]:
        if not v48_lra_generator_enabled:
            return []
        existing = [x for x in existing if isinstance(x, dict)]
        by_label = {str(x.get("candidate_label")): x for x in existing}
        def bundle(base_label: str = "balanced", **updates: Any) -> dict[str, Any]:
            src = copy.deepcopy((by_label.get(base_label) or (existing[0] if existing else {})).get("solution_bundle", {})) if existing else {}
            if not isinstance(src, dict):
                src = {}
            src.update(updates)
            return src
        gap_floor = max(0.0, floor - base_lufs)
        plans = [
            {
                "candidate_label": "v48_limiter_relax",
                "push_db": round(float(np.clip(max(0.35, min(max_push_db, gap_floor - 0.20)), 0.25, max_push_db)), 3),
                "solution_bundle": bundle("balanced", limiter_release_ms=165.0, limiter_lookahead_ms=1.05, clip_drive_scale=0.82, density_recovery_drive_scale=0.82, density_recovery_max_passes=2),
                "intensity": 0.72,
                "v48_candidate_generator": "limiter_relax",
            },
            {
                "candidate_label": "v48_microdyn",
                "push_db": round(float(np.clip(max(0.35, min(max_push_db, gap_floor - 0.35)), 0.25, max_push_db)), 3),
                "solution_bundle": bundle("balanced", limiter_release_ms=155.0, clip_drive_scale=0.76, peak_absorb_drive_scale=0.85, density_recovery_gain_scale=0.82, density_recovery_max_passes=2),
                "intensity": 0.68,
                "v48_candidate_generator": "pre_limiter_microdynamic",
            },
            {
                "candidate_label": "v48_section_safe",
                "push_db": round(float(np.clip(max(0.30, min(max_push_db, gap_floor - 0.55)), 0.25, max_push_db)), 3),
                "solution_bundle": bundle("safe_lift", limiter_release_ms=145.0, clip_drive_scale=0.70, density_recovery_max_passes=1),
                "intensity": 0.62,
                "v48_candidate_generator": "section_aware_lower_push",
            },
        ]
        seen = {round(float((x or {}).get("push_db", 0.0) or 0.0), 2) for x in existing}
        out = []
        for plan in plans:
            if float(plan.get("push_db", 0.0) or 0.0) < 0.18:
                continue
            key = round(float(plan.get("push_db", 0.0) or 0.0), 2)
            if key in seen:
                # Keep the label distinct by nudging down very slightly inside safe bounds.
                plan["push_db"] = round(max(0.18, key - 0.07), 3)
            out.append(plan)
        return out

    # Worker-protection guardrails. These are not musical retry limits; they
    # prevent Cloud Run hangs when a measured blocker loop does not converge.
    safety_max_steps = int(_env_float("BUSY_BLOCKER_RESOLUTION_SAFETY_MAX_STEPS", 18, 4, 48))
    safety_max_rounds_per_start = int(_env_float("BUSY_BLOCKER_RESOLUTION_MAX_ROUNDS_PER_START", 4, 1, 12))
    same_signature_stop_count = int(_env_float("BUSY_BLOCKER_RESOLUTION_SAME_SIGNATURE_STOP_COUNT", 3, 2, 8))
    no_improvement_damage_epsilon = float(_env_float("BUSY_BLOCKER_RESOLUTION_DAMAGE_EPSILON", 0.12, 0.0, 1.0))
    no_improvement_lufs_epsilon = float(_env_float("BUSY_BLOCKER_RESOLUTION_LUFS_EPSILON", 0.08, 0.0, 0.6))
    # Tie the older blocker-loop safety guard to the v34/v35 render budget.
    # This prevents commercial_loudness_finish from rendering 3+ full candidates
    # even when a previous one-shot/legacy planner produces more plans.
    safety_max_steps_unbounded = int(safety_max_steps)
    safety_max_steps = min(int(safety_max_steps), int(commercial_full_render_budget))
    if safety_max_steps < safety_max_steps_unbounded:
        report["optimized_render_budget_enforcement"]["older_safety_max_steps_before_budget_cap"] = int(safety_max_steps_unbounded)
        report["optimized_render_budget_enforcement"]["older_safety_max_steps_after_budget_cap"] = int(safety_max_steps)
    wall_time_limit_sec = float(_env_float("BUSY_BLOCKER_RESOLUTION_WALL_TIME_SEC", 360.0, 30.0, 1500.0))
    loop_started_at = time.monotonic()
    total_steps = 0
    convergence_stops: list[dict[str, Any]] = []
    signature_history: dict[str, int] = {}
    recent_attempt_metrics: list[dict[str, Any]] = []
    report["technical_safety_policy"] = {
        "schema_version": "busy_blocker_resolution_safety_v8_5_3_27",
        "max_total_steps": safety_max_steps,
        "max_rounds_per_start_push": safety_max_rounds_per_start,
        "same_signature_stop_count": same_signature_stop_count,
        "wall_time_limit_sec": round(float(wall_time_limit_sec), 3),
        "damage_epsilon": round(float(no_improvement_damage_epsilon), 3),
        "lufs_epsilon": round(float(no_improvement_lufs_epsilon), 3),
        "policy": "technical worker guard only; musical acceptance still requires deterministic QC safe_checkpoint",
    }

    def _planned_ceiling(plan: dict[str, Any] | None) -> float:
        # v8.5.3.29: Auto Commercial Master is a commercial-reference match path.
        # The private DB loudness ranges are assumed to be referenced closer to
        # hot commercial ceilings, so do not let virtual solver suggestions such
        # as -1.4 dBTP quietly turn the DB target into a streaming-safe target.
        reference_match = bool(profile.get("commercial_reference_match") or _env_bool("BUSY_COMMERCIAL_DB_TARGET_CHASE", True))
        reference_ceiling = float(profile.get("reference_true_peak_ceiling_db", COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB) or COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB)
        planned = reference_ceiling if reference_match else FINAL_TRUE_PEAK_CEILING_DB
        if (not reference_match or _env_bool("BUSY_COMMERCIAL_ALLOW_SOLVER_LOWER_TP", False)) and isinstance(strategy, dict) and strategy.get("recommended_true_peak_ceiling_db") is not None:
            planned = float(np.clip(float(strategy.get("recommended_true_peak_ceiling_db")), -2.0, planned))
        bundle = (plan or {}).get("chosen_bundle", {}) if isinstance((plan or {}).get("chosen_bundle", {}), dict) else {}
        if bundle and (not reference_match or _env_bool("BUSY_COMMERCIAL_ALLOW_BUNDLE_LOWER_TP", False)):
            planned += float(bundle.get("ceiling_delta_db", 0.0) or 0.0)
        return float(np.clip(planned, -2.2, reference_ceiling if reference_match else FINAL_TRUE_PEAK_CEILING_DB))

    def _elapsed_sec() -> float:
        return float(time.monotonic() - loop_started_at)

    def _attempt_damage_score(rep: dict[str, Any]) -> float:
        ps = rep.get("problem_set") if isinstance(rep.get("problem_set"), dict) else {}
        if isinstance(rep.get("acceptance"), dict) and isinstance(rep["acceptance"].get("problem_set"), dict):
            ps = rep["acceptance"].get("problem_set") or ps
        return float(ps.get("damage_score", 0.0) or 0.0)

    def _attempt_lufs(rep: dict[str, Any]) -> float:
        an = rep.get("analysis") if isinstance(rep.get("analysis"), dict) else {}
        return float(an.get("integrated_lufs", an.get("lufs", -99.0)) or -99.0)

    def _attempt_signature(rep: dict[str, Any]) -> str:
        blockers = rep.get("unresolved_blocker_set") or rep.get("hard_blocks") or []
        if not isinstance(blockers, list):
            blockers = []
        return "+".join(sorted({str(x) for x in blockers if str(x).strip()})) or "none"

    def _should_stop_for_stagnation(rep: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        sig = _attempt_signature(rep)
        signature_history[sig] = int(signature_history.get(sig, 0)) + 1
        damage = _attempt_damage_score(rep)
        lufs_now = _attempt_lufs(rep)
        recent_attempt_metrics.append({"signature": sig, "damage_score": damage, "lufs": lufs_now})
        same_recent = [x for x in recent_attempt_metrics[-same_signature_stop_count:] if x.get("signature") == sig]
        if sig != "none" and len(same_recent) >= same_signature_stop_count:
            first = same_recent[0]
            damage_improvement = float(first.get("damage_score", damage) or damage) - damage
            lufs_improvement = lufs_now - float(first.get("lufs", lufs_now) or lufs_now)
            if damage_improvement <= no_improvement_damage_epsilon and lufs_improvement <= no_improvement_lufs_epsilon:
                return True, {
                    "reason": "same_blocker_signature_stagnation_stop",
                    "unresolved_signature": sig,
                    "same_signature_count_total": signature_history.get(sig),
                    "same_signature_count_recent": len(same_recent),
                    "damage_improvement": round(float(damage_improvement), 3),
                    "lufs_improvement_db": round(float(lufs_improvement), 3),
                    "policy": "technical convergence failure; preserve safe checkpoint or base fallback",
                }
        return False, {"unresolved_signature": sig, "same_signature_count_total": signature_history.get(sig)}

    def _render_attempt(push: float, resolution_plan: dict[str, Any] | None, parent_push: float, round_index: int) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], bool, float]:
        planned_ceiling_db = _planned_ceiling(resolution_plan)
        _render_attempt_started_at = time.monotonic()
        conditioned, cond_report = _condition_for_commercial_push(
            y,
            sr,
            before,
            best_analysis,
            push,
            profile,
            virtual_hot_plan=virtual_hot_solver,
            resolution_plan=resolution_plan,
        )
        _debug("commercial_finish_render_condition_done", push_db=round(float(push), 3), elapsed_sec=round(float(time.monotonic() - _render_attempt_started_at), 3), static_moves=cond_report.get("static_move_count"), dynamic_moves=cond_report.get("dynamic_move_count"))
        cond_report["planned_true_peak_ceiling_db"] = round(float(planned_ceiling_db), 3)
        conditioned_analysis = analyze_audio_fast_qc(conditioned, sr, true_peak_oversample=_qc_oversample_factor())
        pre_gain_audio = apply_gain_db(conditioned, push)
        gain_only_analysis = analyze_audio_fast_qc(pre_gain_audio, sr, true_peak_oversample=_qc_oversample_factor())
        _debug("commercial_finish_render_pregain_qc_done", push_db=round(float(push), 3), elapsed_sec=round(float(time.monotonic() - _render_attempt_started_at), 3))
        cond_lufs = float(conditioned_analysis.get("integrated_lufs", base_lufs) or base_lufs)
        gain_lufs = float(gain_only_analysis.get("integrated_lufs", cond_lufs) or cond_lufs)
        gain_delta = gain_lufs - cond_lufs
        cond_report["pre_limiter_gain_application_check"] = {
            "conditioned_lufs": round(float(cond_lufs), 3),
            "gain_only_lufs": round(float(gain_lufs), 3),
            "requested_gain_db": round(float(push), 3),
            "measured_lufs_delta_before_limiter_db": round(float(gain_delta), 3),
            "suspect": bool(float(push) >= 1.0 and float(gain_delta) < max(0.45, float(push) * 0.55)),
            "policy": "gain is checked before clipper/limiter so strategy application bugs are visible",
        }
        render_controls = ((resolution_plan or {}).get("chosen_bundle", {}) if isinstance((resolution_plan or {}).get("chosen_bundle", {}), dict) else {})
        section_hint = render_controls.get("v48_1_section_planner") if isinstance(render_controls.get("v48_1_section_planner"), dict) else {}
        if section_hint and _env_bool("BUSY_V48_1_SECTION_GAIN_ENVELOPE", True):
            pre_gain_audio, section_env_report = apply_section_aware_gain_envelope(pre_gain_audio, sr, section_hint)
            cond_report["v48_1_section_gain_envelope"] = section_env_report
            if section_env_report.get("active"):
                gain_env_analysis = analyze_audio_fast_qc(pre_gain_audio, sr, true_peak_oversample=_qc_oversample_factor())
                cond_report["v48_1_section_gain_envelope"]["post_envelope_analysis"] = _analysis_summary_for_loudness(gain_env_analysis)
        _debug("commercial_finish_hot_render_start", push_db=round(float(push), 3), elapsed_sec=round(float(time.monotonic() - _render_attempt_started_at), 3))
        rendered, hot_render_report = _commercial_hot_candidate_render_with_trace(
            pre_gain_audio,
            sr,
            planned_ceiling_db,
            target,
            floor,
            before,
            decision,
            profile,
            render_controls=render_controls,
        )
        _debug("commercial_finish_hot_render_done", push_db=round(float(push), 3), elapsed_sec=round(float(time.monotonic() - _render_attempt_started_at), 3), trace_len=len(hot_render_report.get("trace", []) if isinstance(hot_render_report, dict) else []))
        cond_report["candidate_render_trace"] = {k: v for k, v in hot_render_report.items() if k != "final_analysis_full"}
        tp_gain = float(hot_render_report.get("initial_limiter_true_peak_normalize_gain_db", 0.0) or 0.0)
        cand_analysis = hot_render_report.get("final_analysis_full") if isinstance(hot_render_report.get("final_analysis_full"), dict) else analyze_audio_fast_qc(rendered, sr, true_peak_oversample=_qc_oversample_factor())
        warnings = _safety_risk(before, cand_analysis, decision)
        hard_blocks = _commercial_loudness_candidate_hard_blocks(before, cand_analysis, decision, profile, warnings)
        lufs = float(cand_analysis.get("integrated_lufs", -99.0) or -99.0)
        crest = float(cand_analysis.get("crest_factor_db", 0.0) or 0.0)
        reached_floor = bool(lufs >= floor)
        reached_target = bool(lufs >= target - 0.20)
        overshot = bool(lufs > target + 0.75 and crest < float(profile.get("crest_min_db", 5.6) or 5.6) + 0.35)
        acceptance = _candidate_acceptance_decision(
            before,
            cand_analysis,
            decision,
            profile,
            warnings + hard_blocks + (["commercial_overpush"] if overshot else []),
            base,
            list(base_problem_report.get("problem_set", []) or []),
            floor,
            target,
        )
        v57_gate = acceptance.get("v57_damage_aware_candidate_gate", {}) if isinstance(acceptance.get("v57_damage_aware_candidate_gate", {}), dict) else {}
        if v57_gate.get("reject"):
            hard_blocks = sorted(set(list(hard_blocks or []) + list(v57_gate.get("blockers", []) or [])))
        hard_blocked = bool(hard_blocks or overshot or not acceptance.get("accepted"))
        score = 0.0
        score += (lufs - base_lufs) * 9.0
        score -= abs(target - lufs) * 3.2
        score += 18.0 if reached_floor else -18.0
        score += 6.0 if reached_target else 0.0
        score += min(7.0, max(0.0, crest - float(profile.get("hard_crest_min_db", 5.0) or 5.0)) * 1.25)
        score -= float(((acceptance.get("problem_set") or {}).get("damage_score", 0.0) or 0.0)) * 3.5
        score -= len(acceptance.get("unresolved_blocker_set", []) or []) * 32.0
        score -= len(warnings) * 2.0
        score -= len(hard_blocks) * 24.0
        score -= float(v57_gate.get("penalty_score", 0.0) or 0.0)
        if overshot:
            score -= 24.0
        rep = {
            "name": f"render_attempt_push_{push:.2f}db_r{round_index}",
            "attempt_type": "render_attempt",
            "parent_start_push_db": round(float(parent_push), 3),
            "push_db": round(float(push), 3),
            "repair_round": int(round_index),
            "true_peak_normalize_gain_db": round(float(tp_gain), 3),
            "conditioning": cond_report,
            "virtual_solver_guided": bool((virtual_hot_solver or {}).get("active")),
            "research_planner_guided": bool(resolution_plan),
            "analysis": _analysis_summary_for_loudness(cand_analysis),
            "warnings": warnings,
            "hard_blocks": hard_blocks + (["commercial_overpush"] if overshot else []),
            "acceptance": acceptance,
            "v57_damage_aware_candidate_gate": v57_gate,
            "problem_set": (acceptance.get("problem_set") or {}),
            "unresolved_blocker_set": sorted(set(acceptance.get("unresolved_blocker_set", []) or hard_blocks or [])),
            "reached_floor": reached_floor,
            "reached_target": reached_target,
            "qc_passed_safe_checkpoint": bool(acceptance.get("accepted") and not hard_blocked),
            "score": round(float(score), 3),
            "candidate_semantics": "this is not a candidate unless qc_passed_safe_checkpoint is true",
            "loop_elapsed_sec": round(float(_elapsed_sec()), 3),
        }
        _attach_pde_to_attempt(rep, cand_analysis)
        return rendered, cand_analysis, rep, hard_blocked, score

    def _score_one_shot_plan_for_full_render_budget(cand_plan: dict[str, Any]) -> dict[str, Any]:
        """Cheap parameter-space preview ranker before any full-track render.

        This is intentionally not an audio render.  It estimates which candidate
        deserves the expensive full render using expected LUFS movement, distance
        to the DB floor/target, blocker-plan safety, and label risk.  The real
        audio-domain QC still decides acceptance after render.
        """
        if not isinstance(cand_plan, dict):
            cand_plan = {}
        label = str(cand_plan.get("candidate_label") or "candidate")
        try:
            push = float(cand_plan.get("push_db", 0.0) or 0.0)
        except Exception:
            push = 0.0
        bundle = cand_plan.get("solution_bundle", {}) if isinstance(cand_plan.get("solution_bundle", {}), dict) else {}
        blocked_moves = bundle.get("blocked_moves", []) if isinstance(bundle.get("blocked_moves", []), list) else []
        ceiling_delta = float(bundle.get("ceiling_delta_db", 0.0) or 0.0)
        clip_scale = float(bundle.get("clip_drive_scale", 1.0) or 1.0)
        low_side = float(bundle.get("low_side_mono_amount", 0.0) or 0.0)
        dyn_count = len(bundle.get("dynamic_moves", []) or []) if isinstance(bundle.get("dynamic_moves", []), list) else 0
        static_count = len(bundle.get("static_moves", []) or []) if isinstance(bundle.get("static_moves", []), list) else 0

        # Full render only needs candidates that can plausibly improve loudness
        # without being obvious overpushes.  Limiters/clip controls often return
        # less than 1:1 LUFS gain, so use a conservative estimate.
        expected_lufs = float(base_lufs) + float(push) * 0.72
        floor_gap_after = max(0.0, float(floor) - expected_lufs)
        target_error = abs(float(target) - expected_lufs)
        score = 50.0
        score += max(0.0, 12.0 - target_error * 3.2)
        score += 9.0 if expected_lufs >= float(floor) else -floor_gap_after * 5.5
        score -= max(0.0, float(push) - max(float(needed), float(gap_to_floor) + 0.35)) * 4.2
        score -= max(0.0, float(push) - float(max_push) * 0.90) * 2.0
        score -= max(0.0, clip_scale - 1.0) * 5.0
        score += max(0.0, -ceiling_delta) * 3.0
        score += min(2.5, low_side * 2.0)
        score += min(2.0, (dyn_count + static_count) * 0.18)
        score -= min(4.5, len(blocked_moves) * 0.35)
        if label in {"safe_lift", "conservative"}:
            score += 1.5
        elif label in {"balanced", "db_floor_chase"}:
            score += 2.4
        elif label in {"db_target_chase", "reference_hot_over", "hot_controlled"}:
            score -= 1.2 + max(0.0, float(push) - float(needed)) * 1.4
        section_adj = section_planner_preview_score_adjustment(cand_plan)
        if abs(section_adj) > 1e-6:
            score += float(section_adj)
        return {
            "candidate_label": label,
            "push_db": round(float(push), 3),
            "preview_score": round(float(score), 3),
            "expected_lufs_proxy": round(float(expected_lufs), 3),
            "floor_gap_after_proxy_db": round(float(floor_gap_after), 3),
            "target_error_proxy_db": round(float(target_error), 3),
            "section_planner_score_adjustment": round(float(section_adj), 3),
            "parameter_space_only": True,
        }

    def _select_one_shot_full_render_candidates(one_shot_candidates: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        scored: list[tuple[float, int, dict[str, Any], dict[str, Any]]] = []
        for idx, cand in enumerate(one_shot_candidates):
            preview = _score_one_shot_plan_for_full_render_budget(cand if isinstance(cand, dict) else {})
            scored.append((float(preview.get("preview_score", -999.0) or -999.0), idx, cand, preview))
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        ranked_preview = [item[3] for item in scored]
        budget = max(1, min(int(commercial_full_render_budget), len(scored)))
        selected = [item[2] for item in scored[:budget]]
        selected_labels = {str((c or {}).get("candidate_label")) for c in selected if isinstance(c, dict)}
        for preview in ranked_preview:
            preview["selected_for_full_render"] = bool(str(preview.get("candidate_label")) in selected_labels)
        return selected, ranked_preview

    if _env_bool("BUSY_BLOCKER_ONE_SHOT_CANDIDATE_SCORER", True) and float(max_push) >= 0.18:
        max_one_shot_candidates = int(_env_float("BUSY_ONE_SHOT_CANDIDATE_MAX", 4, 1, 6))
        if v48_lra_generator_enabled:
            max_one_shot_candidates = max(max_one_shot_candidates, int(_env_float("BUSY_V48_CANDIDATE_MAX", 6, 4, 8)))
        initial_blocker_scores = build_headroom_blocker_scores(before, base, decision, profile, virtual_hot_solver)
        one_shot_plan = build_one_shot_candidate_plan(
            base_lufs=base_lufs,
            target_lufs=target,
            floor_lufs=floor,
            max_push_db=max_push,
            profile=profile,
            strategy=strategy,
            blocker_scores=initial_blocker_scores,
            max_candidates=max_one_shot_candidates,
        )
        if isinstance(decision, dict) and isinstance(decision.get("lgcca_influence_routing", {}), dict):
            one_shot_plan["lgcca_candidate_scoring_context"] = (decision.get("lgcca_influence_routing", {}) or {}).get("candidate_scoring", {})
            one_shot_plan["lgcca_limiter_risk_budget"] = (decision.get("lgcca_influence_routing", {}) or {}).get("limiter_risk_budget", {})
        if isinstance(decision, dict) and isinstance(decision.get("action_governance", {}), dict):
            one_shot_plan["action_governance_context"] = {
                "over_processing_risk_score": (decision.get("action_governance", {}) or {}).get("over_processing_risk_score"),
                "correction_budget": (decision.get("action_governance", {}) or {}).get("correction_budget", {}),
            }
        report["problem_resolution_loop"] = {
            **report.get("problem_resolution_loop", {}),
            "schema_version": "busy_problem_resolution_loop_v8_5_3_29_commercial_reference_target_chase",
            "planner": "one_shot_blocker_scored_candidate_ranker",
            "fixed_retry_limit": True,
            "loop_policy": "blocker scores create a bounded candidate set; no recursive replan; deterministic QC and selection scoring choose one safe checkpoint",
            "failed_attempts_go_to": "render_attempts",
            "qc_passed_attempts_go_to": "safe_checkpoints/candidates",
        }
        report["headroom_blocker_scores_initial"] = initial_blocker_scores
        plan_candidates = list(one_shot_plan.get("candidates", []) or [])
        v48_extra_plans = _v48_extra_candidate_plans(plan_candidates, max_push)
        if v48_extra_plans:
            plan_candidates.extend(v48_extra_plans)
            one_shot_plan["candidates"] = plan_candidates
            one_shot_plan["v48_lra_candidate_generator"] = {
                "enabled": True,
                "active": True,
                "extra_candidate_count": len(v48_extra_plans),
                "extra_labels": [str(x.get("candidate_label")) for x in v48_extra_plans],
                "policy": "add limiter-relax/microdynamic/section-safe candidates so LRA is preserved during candidate generation rather than restored as a destructive post-process",
            }
        section_loudness_planner = {"enabled": False, "reason": "v48_section_planner_disabled"}
        if v48_section_planner_enabled and _env_bool("BUSY_V48_1_SECTION_PLANNER", True):
            try:
                section_loudness_planner = build_section_aware_loudness_planner(
                    y,
                    sr,
                    before,
                    base,
                    decision,
                    profile,
                    virtual_hot_solver,
                )
                if isinstance(section_loudness_planner, dict) and section_loudness_planner.get("active"):
                    plan_candidates, section_candidate_changes = apply_section_planner_to_candidate_plans(plan_candidates, section_loudness_planner)
                    one_shot_plan["candidates"] = plan_candidates
                    section_loudness_planner["candidate_adjustments"] = section_candidate_changes
                    section_loudness_planner["full_render_budget_preserved"] = int(commercial_full_render_budget)
            except Exception as exc:
                section_loudness_planner = {
                    "schema_version": "busy_section_aware_loudness_planner_v8_5_3_48_1",
                    "enabled": True,
                    "active": False,
                    "error": str(exc)[:240],
                    "fallback_policy": "section planner failure does not block mastering; v48.0.1 candidate selection remains available",
                }
        if v48_2_ssrw_enabled and isinstance(ssrw_summary, dict) and ssrw_summary.get("enabled"):
            one_shot_plan["v48_2_stem_sum_residue_witness"] = ssrw_summary
            if isinstance(section_loudness_planner, dict):
                section_loudness_planner.setdefault("ssrw_context", {
                    "global_reliability": ssrw_summary.get("global_reliability"),
                    "action_governance_hints": ssrw_summary.get("action_governance_hints", {}),
                    "policy": "SSRW hints may reduce risky pushes but cannot create extra full renders or override QC.",
                })
        one_shot_plan["v48_1_section_aware_loudness_planner"] = section_loudness_planner
        report["section_aware_loudness_planner"] = section_loudness_planner

        # v8.5.3.54.1: IVRCF Tier-1 judge integration hotfix.  Candidate 1 is seeded
        # from M01-M20 initial_module_state plus Tier-1 judge outputs from
        # structural analysis, dynamics/EQ, limiter pressure and action governance.
        # New strong direct DSP remains staged off; v54 routes the judges through
        # existing solution_bundle controls and existing EQ move lists.
        virtual_mix_conditioning_render = {"enabled": False, "active": False, "reason": "not_run"}
        vmcr_enabled = _env_bool("BUSY_V53_IVRCF_MODULE_STATE_FOUNDATION", _env_bool("BUSY_V52_VIRTUAL_MIX_CONDITIONING_ITERATIVE_RENDER", _env_bool("BUSY_V51_VIRTUAL_MIX_CONDITIONING_ITERATIVE_RENDER", _env_bool("BUSY_V50_VIRTUAL_MIX_CONDITIONING_RENDER", True))))
        # v54.1: BUSY_V54_IVRCF_TIER1_JUDGE_INTEGRATION gates only Tier-1 judge seed,
        # not the whole v53/v52 sequential VMCR/IVRCF loop.
        if vmcr_enabled:
            try:
                plan_candidates, virtual_mix_conditioning_render = expand_candidate_plans_with_virtual_mix_conditioning_feedback(
                    plan_candidates,
                    decision=decision,
                    before_analysis=before,
                    base_analysis=base,
                    profile=profile,
                    section_plan=section_loudness_planner if isinstance(section_loudness_planner, dict) else {},
                    ssrw_summary=ssrw_summary if isinstance(ssrw_summary, dict) else {},
                    target_lufs=target,
                    true_peak_ceiling_db=_planned_ceiling(None),
                    max_iterations=int(_env_float("BUSY_V52_VMCR_MAX_ITERATIONS", 4, 1, 6)),
                    convergence_threshold=float(_env_float("BUSY_V52_VMCR_CONVERGENCE_THRESHOLD", 0.16, 0.06, 0.35)),
                    min_meaningful_improvement=float(_env_float("BUSY_V52_VMCR_MIN_MEANINGFUL_IMPROVEMENT", 0.025, 0.005, 0.12)),
                    apply_ivrcf_initial_controls=_env_bool("BUSY_V53_IVRCF_APPLY_INITIAL_CONTROLS", True),
                    enable_v54_tier1_judges=_env_bool("BUSY_V54_IVRCF_TIER1_JUDGE_INTEGRATION", True),
                )
                one_shot_plan["candidates"] = plan_candidates
            except Exception as exc:
                virtual_mix_conditioning_render = {
                    "schema_version": "busy_virtual_mix_conditioning_iterative_render_v8_5_3_54_1",
                    "enabled": True,
                    "active": False,
                    "reason": "exception_keep_original_candidates",
                    "error": str(exc)[:300],
                }
        one_shot_plan["v54_ivrcf_tier1_judge_integration"] = virtual_mix_conditioning_render
        one_shot_plan["v54_1_ivrcf_tier1_judge_debug_hotfix"] = {
            "supersedes": "v54_ivrcf_tier1_judge_integration",
            **(virtual_mix_conditioning_render if isinstance(virtual_mix_conditioning_render, dict) else {}),
        }
        one_shot_plan["v53_ivrcf_module_state_foundation"] = {
            "superseded_by": "v54_1_ivrcf_tier1_judge_debug_hotfix",
            **(virtual_mix_conditioning_render if isinstance(virtual_mix_conditioning_render, dict) else {}),
        }
        one_shot_plan["v52_virtual_mix_conditioning_iterative_render"] = {
            "superseded_by": "v54_1_ivrcf_tier1_judge_debug_hotfix",
            **(virtual_mix_conditioning_render if isinstance(virtual_mix_conditioning_render, dict) else {}),
        }
        one_shot_plan["v51_virtual_mix_conditioning_iterative_render"] = {
            "superseded_by": "v54_1_ivrcf_tier1_judge_debug_hotfix",
            **(virtual_mix_conditioning_render if isinstance(virtual_mix_conditioning_render, dict) else {}),
        }
        one_shot_plan["v50_virtual_mix_conditioning_render"] = {
            "superseded_by": "v54_1_ivrcf_tier1_judge_debug_hotfix",
            **(virtual_mix_conditioning_render if isinstance(virtual_mix_conditioning_render, dict) else {}),
        }
        report["virtual_mix_conditioning_render"] = virtual_mix_conditioning_render
        full_render_candidates, preview_ranked_candidates = _select_one_shot_full_render_candidates(plan_candidates)
        if v48_extra_plans and full_render_candidates:
            budget = max(1, min(int(commercial_full_render_budget), len(plan_candidates)))
            labels_selected = {str((c or {}).get("candidate_label")) for c in full_render_candidates if isinstance(c, dict)}
            # Ensure at least one v48 LRA-preserving candidate reaches real render.
            if (not v48_2_2_virtual_proxy_enabled) and not any(str((x or {}).get("candidate_label")) in labels_selected for x in v48_extra_plans):
                full_render_candidates = (full_render_candidates[: max(0, budget - 1)] + [v48_extra_plans[0]])[:budget]
                one_shot_plan.setdefault("v48_lra_candidate_generator", {})["forced_one_candidate_into_full_render"] = True

        # v48.2.2: replace repeated full candidate renders with full-track
        # virtual proxy ranking.  All recipe candidates are scored using cheap
        # whole-song metrics; only the selected proxy winner proceeds to a real
        # full render.  This is not a musical exception cap: it is a render-path
        # change from try-render-reject loops to parameter-space simulation.
        virtual_proxy_plan = {"enabled": bool(v48_2_2_virtual_proxy_enabled), "active": False, "reason": "disabled_by_env"}
        if v48_2_2_virtual_proxy_enabled:
            try:
                virtual_proxy_plan = build_virtual_proxy_candidate_plan(
                    base_analysis=base,
                    before_analysis=before,
                    candidate_plans=plan_candidates,
                    decision=decision,
                    profile=profile,
                    ssrw_summary=ssrw_summary if isinstance(ssrw_summary, dict) else {},
                    section_plan=section_loudness_planner if isinstance(section_loudness_planner, dict) else {},
                    target_lufs=target,
                    floor_lufs=floor,
                    true_peak_ceiling_db=_planned_ceiling(None),
                    max_full_renders=int(_env_float("BUSY_V48_2_2_MAX_SELECTED_FULL_RENDERS", 1, 1, 2)),
                )
                if isinstance(virtual_proxy_plan, dict) and virtual_proxy_plan.get("active"):
                    selected_proxy_plans = [p for p in virtual_proxy_plan.get("selected_candidate_plans", []) if isinstance(p, dict)]
                    if selected_proxy_plans:
                        full_render_candidates = selected_proxy_plans
                        selected_labels = {str((c or {}).get("candidate_label")) for c in full_render_candidates if isinstance(c, dict)}
                        for preview in preview_ranked_candidates:
                            preview["selected_for_full_render_before_virtual_proxy"] = bool(preview.get("selected_for_full_render"))
                            preview["selected_for_full_render"] = bool(str(preview.get("candidate_label")) in selected_labels)
                        one_shot_plan["virtual_proxy_selected_only_full_render"] = True
                    else:
                        virtual_proxy_plan["fallback_reason"] = "no_selected_proxy_plan_keep_original_preview_selection"
            except Exception as exc:
                virtual_proxy_plan = {
                    "schema_version": "busy_virtual_proxy_render_plan_v8_5_3_48_2_2",
                    "enabled": True,
                    "active": False,
                    "reason": "exception_keep_original_preview_selection",
                    "error": str(exc)[:300],
                }
        one_shot_plan["v48_2_2_virtual_proxy_render"] = {k: v for k, v in virtual_proxy_plan.items() if k != "selected_candidate_plans"}
        report["virtual_proxy_render"] = one_shot_plan["v48_2_2_virtual_proxy_render"]
        one_shot_plan["preview_ranked_candidates"] = preview_ranked_candidates

        # v8.5.3.37: reserve Gemini for a final-limited adaptive preview
        # master instead of spending it on the raw A-1 preview. The shared
        # adaptive planner now chooses 20-40s of high-information windows.
        # These are rendered only as a short in-memory preview through the
        # current top candidate chain and final limiter, then only bounded
        # corrections are allowed before the actual full-track render budget is
        # spent. Missing API key/table/history does not block deterministic
        # mastering.
        preview_master_review: dict[str, Any] = {"enabled": False, "reason": "not_run"}
        gemini_preview_allowed = bool(
            full_render_candidates
            and _env_bool("BUSY_GEMINI_PREVIEW_MASTER_REVIEW", False)
            and not (v48_2_2_virtual_proxy_enabled and _env_bool("BUSY_V48_2_2_SKIP_GEMINI_PREVIEW_REVIEW", True))
        )
        if gemini_preview_allowed:
            try:
                top_plan = full_render_candidates[0] if isinstance(full_render_candidates[0], dict) else {}
                preview_master_adaptive_plan = build_adaptive_audio_preview_plan(
                    duration_sec=float(len(y) / max(int(sr), 1)) if y is not None else 0.0,
                    features={
                        **(before if isinstance(before, dict) else {}),
                        "decision": decision if isinstance(decision, dict) else {},
                        "profile": profile,
                        "auto_intensity_policy": (decision.get("auto_intensity_policy") if isinstance(decision, dict) else {}),
                    },
                    segment_analysis=((before or {}).get("segment_analysis") if isinstance(before, dict) else {}),
                    purpose="gemini_preview_master_final_limiter_review",
                    min_total_sec=float(_env_float("BUSY_GEMINI_PREVIEW_MASTER_MIN_SEC", 20.0, 8.0, 40.0)),
                    base_total_sec=float(_env_float("BUSY_GEMINI_PREVIEW_MASTER_BASE_SEC", _env_float("BUSY_GEMINI_PREVIEW_MASTER_TOTAL_SEC", 32.0, 12.0, 40.0), 12.0, 40.0)),
                    max_total_sec=float(_env_float("BUSY_GEMINI_PREVIEW_MASTER_MAX_SEC", 40.0, 12.0, 40.0)),
                    default_segment_sec=float(_env_float("BUSY_GEMINI_PREVIEW_MASTER_SEGMENT_SEC", 8.0, 3.0, 12.0)),
                    max_segments=int(_env_float("BUSY_GEMINI_PREVIEW_MASTER_MAX_SEGMENTS", _env_float("BUSY_GEMINI_PREVIEW_MASTER_SEGMENTS", 5, 1, 6), 1, 6)),
                ) if _env_bool("BUSY_GEMINI_PREVIEW_MASTER_ADAPTIVE", True) else {"enabled": False, "reason": "BUSY_GEMINI_PREVIEW_MASTER_ADAPTIVE disabled", "schema_version": "busy_adaptive_audio_preview_plan_v8_5_3_37"}
                preview_source, preview_source_summary = _build_preview_master_source_audio(
                    y,
                    sr,
                    before if isinstance(before, dict) else {},
                    total_sec=float(_env_float("BUSY_GEMINI_PREVIEW_MASTER_TOTAL_SEC", 32.0, 12.0, 40.0)),
                    segment_sec=float(_env_float("BUSY_GEMINI_PREVIEW_MASTER_SEGMENT_SEC", 8.0, 3.0, 12.0)),
                    max_segments=int(_env_float("BUSY_GEMINI_PREVIEW_MASTER_SEGMENTS", 5, 1, 6)),
                    adaptive_plan=preview_master_adaptive_plan,
                )
                top_push = float(top_plan.get("push_db", 0.0) or 0.0)
                top_bundle = top_plan.get("solution_bundle", {}) if isinstance(top_plan.get("solution_bundle", {}), dict) else {}
                preview_resolution_plan = {
                    "active": True,
                    "schema_version": "busy_preview_master_resolution_plan_v8_5_3_37_adaptive",
                    "planner": "gemini_preview_master_pre_full_render",
                    "candidate_label": str(top_plan.get("candidate_label") or "candidate"),
                    "chosen_bundle": top_bundle,
                    "policy": "short-window final-limited preview only; not persisted as final WAV",
                }
                preview_ceiling = _planned_ceiling(preview_resolution_plan)
                preview_conditioned, preview_cond_report = _condition_for_commercial_push(
                    preview_source,
                    sr,
                    before if isinstance(before, dict) else {},
                    base,
                    top_push,
                    profile,
                    virtual_hot_plan=virtual_hot_solver,
                    resolution_plan=preview_resolution_plan,
                )
                preview_pre_gain = apply_gain_db(preview_conditioned, top_push)
                preview_rendered, preview_render_report = _commercial_hot_candidate_render_with_trace(
                    preview_pre_gain,
                    sr,
                    preview_ceiling,
                    target,
                    floor,
                    before if isinstance(before, dict) else {},
                    decision,
                    profile,
                    render_controls=top_bundle,
                )
                preview_analysis = preview_render_report.get("final_analysis_full") if isinstance(preview_render_report.get("final_analysis_full"), dict) else analyze_audio_fast_qc(preview_rendered, sr, true_peak_oversample=_qc_oversample_factor())
                encoded_preview = _encode_preview_master_wav_base64(
                    preview_rendered,
                    sr,
                    target_sr=int(_env_float("BUSY_GEMINI_PREVIEW_MASTER_SR", 24000, 16000, 48000)),
                )
                preview_summary = {k: v for k, v in encoded_preview.items() if k != "base64"}
                preview_summary.update(preview_source_summary)
                preview_summary["adaptive_preview_plan"] = preview_master_adaptive_plan
                candidate_context = {
                    "candidate_label": str(top_plan.get("candidate_label") or "candidate"),
                    "push_db": round(float(top_push), 3),
                    "planned_true_peak_ceiling_db": round(float(preview_ceiling), 3),
                    "preview_analysis": _analysis_summary_for_loudness(preview_analysis),
                    "preview_conditioning_report": {k: v for k, v in preview_cond_report.items() if k != "research_driven_blocker_solution_planner"},
                    "preview_render_trace": {k: v for k, v in preview_render_report.items() if k != "final_analysis_full"},
                    "full_render_budget": int(commercial_full_render_budget),
                }
                try:
                    from ai_decision.gemini_audio_judge import run_gemini_preview_master_review
                    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("BUSY_GEMINI_API_KEY")
                    preview_master_review = run_gemini_preview_master_review(
                        api_key=gemini_key,
                        preview_wav_base64=encoded_preview.get("base64"),
                        preview_summary=preview_summary,
                        candidate_context=candidate_context,
                        features=before if isinstance(before, dict) else {},
                        decision=decision,
                        model=os.environ.get("BUSY_GEMINI_PREVIEW_MASTER_MODEL", os.environ.get("BUSY_GEMINI_AUDIO_MODEL", "gemini-3.1-flash-lite")) or "gemini-3.1-flash-lite",
                        mode=mode,
                    )
                except Exception as review_exc:
                    preview_master_review = {"available": False, "reason": "gemini_preview_master_review_exception", "error": str(review_exc)[:500], "schema_version": "busy_gemini_preview_master_review_v8_5_3_37_adaptive"}
                preview_master_review["preview_summary"] = preview_summary
                preview_master_review["candidate_context"] = candidate_context
                if _preview_master_review_should_correct(preview_master_review):
                    full_render_candidates = [_apply_preview_master_review_to_candidate(c, preview_master_review) for c in full_render_candidates if isinstance(c, dict)]
                    preview_master_review["bounded_correction_applied_to_full_render_candidates"] = True
                else:
                    preview_master_review["bounded_correction_applied_to_full_render_candidates"] = False
            except Exception as preview_exc:
                preview_master_review = {"enabled": True, "available": False, "reason": "preview_master_review_preparation_failed", "error": str(preview_exc)[:500], "schema_version": "busy_gemini_preview_master_review_v8_5_3_37_adaptive"}
        elif not full_render_candidates:
            preview_master_review = {"enabled": False, "reason": "no_full_render_candidates"}
        else:
            preview_master_review = {"enabled": False, "reason": "BUSY_GEMINI_PREVIEW_MASTER_REVIEW disabled"}
        one_shot_plan["gemini_preview_master_review"] = {k: v for k, v in preview_master_review.items() if k not in {"raw_text"}}
        report["preview_master_audio_review"] = one_shot_plan["gemini_preview_master_review"]

        one_shot_plan["full_render_candidates"] = [
            {
                "candidate_label": str((c or {}).get("candidate_label") or "candidate"),
                "push_db": round(float((c or {}).get("push_db", 0.0) or 0.0), 3),
                **({"push_db_before_v48_1_section_planner": (c or {}).get("push_db_before_v48_1_section_planner")} if isinstance(c, dict) and (c or {}).get("push_db_before_v48_1_section_planner") is not None else {}),
                **({"v48_1_section_planner": (c or {}).get("v48_1_section_planner")} if isinstance(c, dict) and (c or {}).get("v48_1_section_planner") else {}),
                **({"push_db_before_gemini_preview_review": (c or {}).get("push_db_before_gemini_preview_review")} if isinstance(c, dict) and (c or {}).get("push_db_before_gemini_preview_review") is not None else {}),
                **({"gemini_preview_master_correction_applied": (c or {}).get("gemini_preview_master_correction_applied")} if isinstance(c, dict) and (c or {}).get("gemini_preview_master_correction_applied") else {}),
                **({"virtual_mix_conditioning_feedback": (c or {}).get("virtual_mix_conditioning_feedback")} if isinstance(c, dict) and (c or {}).get("virtual_mix_conditioning_feedback") else {}),
                **({"virtual_mix_conditioning_iteration": (c or {}).get("virtual_mix_conditioning_iteration")} if isinstance(c, dict) and (c or {}).get("virtual_mix_conditioning_iteration") else {}),
                **({"virtual_mix_conditioning_iteration_trace": (c or {}).get("virtual_mix_conditioning_iteration_trace")} if isinstance(c, dict) and (c or {}).get("virtual_mix_conditioning_iteration_trace") else {}),
                **({"virtual_mix_conditioning_context": (c or {}).get("virtual_mix_conditioning_render_context")} if isinstance(c, dict) and (c or {}).get("virtual_mix_conditioning_render_context") else {}),
            }
            for c in full_render_candidates if isinstance(c, dict)
        ]
        one_shot_plan["full_render_budget_enforced"] = {
            "active": True,
            "plan_candidate_count": len(plan_candidates),
            "full_render_candidate_count": len(full_render_candidates),
            "full_render_budget": int(commercial_full_render_budget),
            "dropped_before_full_render_count": max(0, len(plan_candidates) - len(full_render_candidates)),
            "policy": "parameter-space preview ranking prunes candidate plans before expensive full-track render",
        }
        report["one_shot_candidate_plan"] = one_shot_plan
        report["candidate_selection_policy"] = "v8.5.3.35: parameter-space preview ranks candidates first; only the top one/two are allowed to full-render; only deterministic-QC-passed safe_checkpoints can be selected"
        _debug(
            "commercial_finish_one_shot_plan",
            candidate_count=len(plan_candidates),
            full_render_candidate_count=len(full_render_candidates),
            full_render_budget=int(commercial_full_render_budget),
            top_blockers=[b.get("name") for b in initial_blocker_scores[:4]],
        )

        for cand_plan in full_render_candidates:
            if total_steps >= safety_max_steps:
                convergence_stops.append({"reason": "technical_safety_max_steps_reached_one_shot", "max_steps": safety_max_steps})
                break
            if _elapsed_sec() > wall_time_limit_sec:
                convergence_stops.append({
                    "reason": "technical_wall_time_limit_reached_one_shot",
                    "elapsed_sec": round(float(_elapsed_sec()), 3),
                    "wall_time_limit_sec": round(float(wall_time_limit_sec), 3),
                })
                break
            total_steps += 1
            label = str(cand_plan.get("candidate_label") or f"candidate_{total_steps}")
            candidate_push = float(cand_plan.get("push_db", 0.0) or 0.0)
            _v6314_push_cap = _v6314_repeated_crest_loss_push_cap(report, candidate_push, decision)
            if _v6314_push_cap.get("active"):
                candidate_push = float(_v6314_push_cap.get("effective_push_db", candidate_push) or candidate_push)
                cand_plan["v6314_repeated_crest_loss_push_cap"] = _v6314_push_cap
            candidate_bundle = cand_plan.get("solution_bundle", {}) if isinstance(cand_plan.get("solution_bundle", {}), dict) else {}
            resolution_plan = {
                "active": True,
                "schema_version": "busy_one_shot_resolution_plan_v8_5_3_27",
                "planner": "one_shot_blocker_scored_candidate_ranker",
                "candidate_label": label,
                "chosen_bundle": candidate_bundle,
                "initial_headroom_blocker_scores": initial_blocker_scores,
                "convergence": {"stop": True, "stop_reason": "one_shot_candidate_no_recursive_replan"},
                "policy": "candidate applies one bounded solution_bundle once; failure does not spawn another loop",
            }
            try:
                try:
                    gc.collect()
                except Exception:
                    pass
                _debug("commercial_finish_one_shot_render_attempt_start", label=label, push_db=round(float(candidate_push), 3), loop_elapsed_sec=round(float(_elapsed_sec()), 3))
                cand_audio, cand_analysis, rep, hard_blocked, _old_score = _render_attempt(candidate_push, resolution_plan, candidate_push, 1)
                after_blocker_scores = build_headroom_blocker_scores(before, cand_analysis, decision, profile, virtual_hot_solver)
                selection_score = score_loudness_candidate(
                    base_analysis=base,
                    candidate_analysis=cand_analysis,
                    before_blockers=initial_blocker_scores,
                    after_blockers=after_blocker_scores,
                    acceptance=rep.get("acceptance", {}) if isinstance(rep.get("acceptance", {}), dict) else {},
                    warnings=rep.get("warnings", []) if isinstance(rep.get("warnings", []), list) else [],
                    hard_blocks=rep.get("hard_blocks", []) if isinstance(rep.get("hard_blocks", []), list) else [],
                    profile=profile,
                    target_lufs=target,
                    floor_lufs=floor,
                )
                rep["name"] = f"render_attempt_{label}_{candidate_push:.2f}db"
                rep["candidate_label"] = label
                rep["attempt_type"] = "render_attempt"
                rep["one_shot_candidate_ranker"] = {
                    "active": True,
                    "candidate_plan": cand_plan,
                    "headroom_blocker_scores_after": after_blocker_scores,
                    "selection_score": selection_score,
                    "no_recursive_replan": True,
                }
                v57_rep_gate = rep.get("v57_damage_aware_candidate_gate", {}) if isinstance(rep.get("v57_damage_aware_candidate_gate", {}), dict) else {}
                if v57_rep_gate:
                    selection_score["v57_damage_penalty"] = float(v57_rep_gate.get("penalty_score", 0.0) or 0.0)
                    selection_score["v57_accepted_by_damage_gate"] = bool(not v57_rep_gate.get("reject"))
                rep["selection_score"] = selection_score
                rep["score"] = round(float(selection_score.get("score_0_100", -999.0) or -999.0) - float(v57_rep_gate.get("penalty_score", 0.0) or 0.0), 3)
                rep["candidate_semantics"] = "this render_attempt becomes a candidate only if qc_passed_safe_checkpoint is true"
                report["render_attempts"].append(rep)
                _debug("commercial_finish_one_shot_render_attempt", **{k: v for k, v in rep.items() if k not in {"conditioning", "analysis", "acceptance", "problem_set", "one_shot_candidate_ranker"}})

                lufs = float(cand_analysis.get("integrated_lufs", -99.0) or -99.0)
                v6314_dyn = _v6314_candidate_dynamic_governance(best_analysis, cand_analysis, decision, floor, candidate_label=label)
                rep["v6314_candidate_dynamic_governance"] = v6314_dyn
                if v6314_dyn.get("reject_for_cleaner_quieter"):
                    rep["qc_passed_safe_checkpoint_before_v6314"] = bool(rep.get("qc_passed_safe_checkpoint"))
                    rep["qc_passed_safe_checkpoint"] = False
                    rep.setdefault("unresolved_blocker_set", [])
                    if isinstance(rep.get("unresolved_blocker_set"), list) and "v6314_cleaner_quieter_reference_preferred" not in rep["unresolved_blocker_set"]:
                        rep["unresolved_blocker_set"].append("v6314_cleaner_quieter_reference_preferred")
                if rep.get("qc_passed_safe_checkpoint") and lufs > base_lufs + 0.18:
                    checkpoint = copy.deepcopy(rep)
                    checkpoint["attempt_type"] = "safe_checkpoint"
                    checkpoint["name"] = f"safe_checkpoint_{label}_{candidate_push:.2f}db"
                    checkpoint["selected_eligibility"] = "eligible_only_because_deterministic_qc_passed_and_selection_score_available"
                    checkpoint["safe_checkpoint_policy"] = "ranked by loudness/quality/profile-fit/blocker-reduction/safety-margin score"
                    report["safe_checkpoints"].append(checkpoint)
                    report["candidates"].append(checkpoint)
                    score_now = float(rep.get("score", selection_score.get("score_0_100", -999.0)) or -999.0)
                    if score_now > best_score:
                        best_audio = cand_audio
                        best_analysis = cand_analysis
                        best_score = score_now
                        best_name = checkpoint["name"]
                    if _env_bool("BUSY_COMMERCIAL_FINISH_STOP_AFTER_SAFE_CHECKPOINT", True) and (lufs >= floor or int(commercial_full_render_budget) <= len(report.get("render_attempts", []) or [])):
                        convergence_stops.append({
                            "candidate_label": label,
                            "push_db": round(float(candidate_push), 3),
                            "reason": "decision_convergence_safe_checkpoint_early_exit",
                            "reached_floor": bool(lufs >= floor),
                            "render_attempt_count": len(report.get("render_attempts", []) or []),
                            "full_render_budget": int(commercial_full_render_budget),
                        })
                        break
                else:
                    repair_audio, repair_analysis, repair_rep = _repair_nearmiss_candidate(cand_audio, cand_analysis, rep, label=label, push_db=candidate_push)
                    if isinstance(repair_rep, dict):
                        report.setdefault("nearmiss_events", []).append({k: v for k, v in repair_rep.items() if k not in {"acceptance", "problem_set"}})
                        if repair_rep.get("accepted") and isinstance(repair_audio, np.ndarray) and isinstance(repair_analysis, dict):
                            after_blocker_scores_repaired = build_headroom_blocker_scores(before, repair_analysis, decision, profile, virtual_hot_solver)
                            repair_score = score_loudness_candidate(
                                base_analysis=base,
                                candidate_analysis=repair_analysis,
                                before_blockers=initial_blocker_scores,
                                after_blockers=after_blocker_scores_repaired,
                                acceptance=repair_rep.get("acceptance", {}) if isinstance(repair_rep.get("acceptance", {}), dict) else {},
                                warnings=repair_rep.get("warnings", []) if isinstance(repair_rep.get("warnings", []), list) else [],
                                hard_blocks=repair_rep.get("hard_blocks", []) if isinstance(repair_rep.get("hard_blocks", []), list) else [],
                                profile=profile,
                                target_lufs=target,
                                floor_lufs=floor,
                            )
                            repair_gate = _v57_damage_aware_candidate_gate(before, repair_analysis, base, decision, profile, repair_rep.get("warnings", []) if isinstance(repair_rep.get("warnings", []), list) else [], floor, target)
                            repair_rep["v57_damage_aware_candidate_gate"] = repair_gate
                            repair_score["v57_damage_penalty"] = float(repair_gate.get("penalty_score", 0.0) or 0.0)
                            if repair_gate.get("reject"):
                                repair_rep["accepted"] = False
                                repair_rep["qc_passed_safe_checkpoint"] = False
                                repair_rep["v57_rejected_after_nearmiss_repair"] = True
                                repair_rep["unresolved_blocker_set"] = sorted(set((repair_rep.get("unresolved_blocker_set", []) or []) + (repair_gate.get("blockers", []) or [])))
                            repair_rep["selection_score"] = repair_score
                            repair_rep["score"] = round(float(repair_score.get("score_0_100", -999.0) or -999.0) - float(repair_gate.get("penalty_score", 0.0) or 0.0), 3)
                            if repair_gate.get("reject"):
                                report["render_attempts"].append(repair_rep)
                                convergence_stops.append({
                                    "candidate_label": label,
                                    "push_db": round(float(candidate_push), 3),
                                    "reason": "v57_rejected_nearmiss_repair_damage_gate",
                                    "v57_blockers": repair_gate.get("blockers", []),
                                })
                                continue
                            checkpoint = copy.deepcopy(repair_rep)
                            checkpoint["attempt_type"] = "safe_checkpoint"
                            checkpoint["name"] = f"safe_checkpoint_nearmiss_repair_{label}_{candidate_push:.2f}db"
                            checkpoint["selected_eligibility"] = "eligible_after_single_pass_nearmiss_repair_and_full_reanalysis"
                            report["render_attempts"].append(repair_rep)
                            report["safe_checkpoints"].append(checkpoint)
                            report["candidates"].append(checkpoint)
                            score_now = float(repair_score.get("score_0_100", -999.0) or -999.0)
                            if score_now > best_score:
                                best_audio = repair_audio
                                best_analysis = repair_analysis
                                best_score = score_now
                                best_name = checkpoint["name"]
                            convergence_stops.append({
                                "candidate_label": label,
                                "push_db": round(float(candidate_push), 3),
                                "reason": "v48_nearmiss_repair_created_safe_checkpoint",
                                "repaired_checkpoint": checkpoint["name"],
                            })
                            if _env_bool("BUSY_V48_STOP_AFTER_NEARMISS_SAFE", True):
                                break
                        elif repair_rep.get("enabled", True) and repair_rep.get("attempt_type") == "nearmiss_repair" and repair_rep.get("name"):
                            report["render_attempts"].append(repair_rep)
                    convergence_stops.append({
                        "candidate_label": label,
                        "push_db": round(float(candidate_push), 3),
                        "reason": "one_shot_candidate_failed_qc_or_not_louder_than_base",
                        "unresolved_blocker_set": rep.get("unresolved_blocker_set", []),
                    })
            except Exception as exc:
                err = {
                    "name": f"render_attempt_{label}_{candidate_push:.2f}db",
                    "attempt_type": "render_attempt",
                    "candidate_label": label,
                    "error": str(exc)[:300],
                    "one_shot_candidate_ranker": {"active": True, "failed_before_qc": True},
                }
                report["render_attempts"].append(err)
                _debug("commercial_finish_one_shot_render_attempt_failed", label=label, push_db=round(float(candidate_push), 3), error=str(exc)[:300])

        best_lufs = float(best_analysis.get("integrated_lufs", base_lufs) or base_lufs)
        selected = best_name != "base"
        report["selected"] = best_name
        report["selected_score"] = round(float(best_score), 3) if selected else None
        report["output"] = _analysis_summary_for_loudness(best_analysis)
        report["lufs_gain_achieved_db"] = round(float(best_lufs - base_lufs), 3)
        report["target_met"] = bool(best_lufs >= floor)
        report["target_reached"] = bool(best_lufs >= target - 0.20)
        report["safe_checkpoint_count"] = len(report.get("safe_checkpoints", []) or [])
        report["render_attempt_count"] = len(report.get("render_attempts", []) or [])
        report["convergence_stops"] = convergence_stops
        report["technical_safety_summary"] = {
            "elapsed_sec": round(float(_elapsed_sec()), 3),
            "total_steps": int(total_steps),
            "candidate_count": len(one_shot_plan.get("candidates", []) or []),
            "full_render_candidate_count": len(one_shot_plan.get("full_render_candidates", []) or []),
            "full_render_budget": int(commercial_full_render_budget),
            "dropped_before_full_render_count": int((one_shot_plan.get("full_render_budget_enforced", {}) or {}).get("dropped_before_full_render_count", 0) or 0),
            "hit_wall_time_limit": bool(_elapsed_sec() > wall_time_limit_sec),
            "hit_total_step_limit": bool(total_steps >= safety_max_steps),
            "safe_checkpoint_available": bool(report.get("safe_checkpoints")),
            "fallback_policy": "selected ranked safe_checkpoint when available, otherwise base fallback",
        }
        report["adaptive_candidate_decision_convergence"] = {
            "schema_version": "busy_adaptive_candidate_decision_convergence_v8_5_3_48_2_2_virtual_proxy",
            "active": True,
            "render_count_is_safety_cap_not_musical_rule": True,
            "safe_checkpoint_count": len(report.get("safe_checkpoints", []) or []),
            "render_attempt_count": len(report.get("render_attempts", []) or []),
            "convergence_stops": convergence_stops,
            "ai_reallocation": {
                "realtime": "candidate_comparison_only_if_top_candidates_close",
                "gemini": "final_quality_validation_only_if_nearmiss_or_risk_flag",
                "gpt54mini": "summary_public_safe_explanation_and_gpt55_packet",
                "gpt55": "hard_case_arbitration_within_allowed_candidates",
                "hard_qc_is_law": True,
                "no_ai_dsp_control": True,
            },
            "ai_call_acceleration_policy": "AI calls are allowed to shorten candidate loops by confirming convergence or arbitration, not to bypass hard vetoes or add unrestricted renders.",
        }
        report["selected_master_policy"] = "select highest-scoring measured safe checkpoint from bounded convergence candidates; if none exists, base remains selected"
        if report["target_met"]:
            report["reason"] = "commercial_floor_reached_after_one_shot_candidate_ranking" if selected else "commercial_floor_already_met"
        elif selected:
            report["reason"] = "best_ranked_safe_checkpoint_selected_but_commercial_floor_not_reached"
            report["retry_blocked_by"] = "one_shot_candidate_set_exhausted_before_floor"
        else:
            report["reason"] = "commercial_floor_not_reached_no_safe_checkpoint"
            blocks: list[str] = []
            for attempt in report.get("render_attempts", []):
                if isinstance(attempt, dict):
                    blocks.extend(attempt.get("unresolved_blocker_set", []) or attempt.get("hard_blocks", []) or [])
            report["retry_blocked_by"] = sorted(set(blocks)) or "no_qc_safe_louder_checkpoint"
        if isinstance(report.get("v57_commercial_loudness_feasibility"), dict) and report["v57_commercial_loudness_feasibility"].get("decision") == "do_not_push_select_best_safe_candidate" and not selected:
            report["reason"] = "v57_no_safe_push_budget_select_base_or_existing_safe_candidate"
            report["retry_blocked_by"] = report["v57_commercial_loudness_feasibility"].get("reasons")
        report["technical_safety_guard"] = {
            "hit": bool(total_steps >= safety_max_steps or _elapsed_sec() > wall_time_limit_sec),
            "max_steps": safety_max_steps,
            "wall_time_limit_sec": round(float(wall_time_limit_sec), 3),
            "policy": "one-shot candidate ranker avoids recursive blocker loops by design",
        }
        _v57_attach_damage_aware_selection_summary(report)
        _attach_loudness_checkpoint_diversity_report(report)
        return np.nan_to_num(best_audio, nan=0.0, posinf=0.0, neginf=0.0), report

    for start_push in pushes:
        if total_steps >= safety_max_steps:
            convergence_stops.append({"start_push_db": round(float(start_push), 3), "reason": "technical_safety_max_steps_reached"})
            break
        current_push = float(start_push)
        resolution_plan: dict[str, Any] | None = None
        local_failed_attempts: list[dict[str, Any]] = []
        round_index = 0
        while total_steps < safety_max_steps:
            if round_index >= safety_max_rounds_per_start:
                convergence_stops.append({
                    "start_push_db": round(float(start_push), 3),
                    "last_push_db": round(float(current_push), 3),
                    "reason": "technical_max_rounds_per_start_reached",
                    "max_rounds_per_start": safety_max_rounds_per_start,
                })
                break
            if _elapsed_sec() > wall_time_limit_sec:
                convergence_stops.append({
                    "start_push_db": round(float(start_push), 3),
                    "last_push_db": round(float(current_push), 3),
                    "reason": "technical_wall_time_limit_reached",
                    "elapsed_sec": round(float(_elapsed_sec()), 3),
                    "wall_time_limit_sec": round(float(wall_time_limit_sec), 3),
                })
                break
            round_index += 1
            total_steps += 1
            try:
                cand_audio, cand_analysis, rep, hard_blocked, score = _render_attempt(current_push, resolution_plan, start_push, round_index)
                report["render_attempts"].append(rep)
                _debug("commercial_finish_render_attempt", **{k: v for k, v in rep.items() if k not in {"conditioning", "analysis", "acceptance", "problem_set"}})

                stagnated, stagnation_report = _should_stop_for_stagnation(rep)
                rep["stagnation_guard"] = stagnation_report

                v6314_dyn = _v6314_candidate_dynamic_governance(best_analysis, cand_analysis, decision, floor, candidate_label=f"push_{current_push:.2f}_r{round_index}")
                rep["v6314_candidate_dynamic_governance"] = v6314_dyn
                if v6314_dyn.get("reject_for_cleaner_quieter"):
                    rep["qc_passed_safe_checkpoint_before_v6314"] = bool(rep.get("qc_passed_safe_checkpoint"))
                    rep["qc_passed_safe_checkpoint"] = False
                    rep.setdefault("unresolved_blocker_set", [])
                    if isinstance(rep.get("unresolved_blocker_set"), list) and "v6314_cleaner_quieter_reference_preferred" not in rep["unresolved_blocker_set"]:
                        rep["unresolved_blocker_set"].append("v6314_cleaner_quieter_reference_preferred")
                if rep.get("qc_passed_safe_checkpoint") and float(cand_analysis.get("integrated_lufs", -99.0) or -99.0) > base_lufs + 0.18:
                    checkpoint = copy.deepcopy(rep)
                    checkpoint["attempt_type"] = "safe_checkpoint"
                    checkpoint["name"] = f"safe_checkpoint_push_{current_push:.2f}db_r{round_index}"
                    checkpoint["selected_eligibility"] = "eligible_only_because_qc_passed_no_unresolved_new_problem_set"
                    report["safe_checkpoints"].append(checkpoint)
                    report["candidates"].append(checkpoint)
                    if score > best_score:
                        best_audio = cand_audio
                        best_analysis = cand_analysis
                        best_score = score
                        best_name = checkpoint["name"]
                    # A safe checkpoint at this start push is enough; higher start
                    # pushes can still be explored by the outer schedule.
                    break

                if stagnated:
                    convergence_stops.append({
                        "start_push_db": round(float(start_push), 3),
                        "last_push_db": round(float(current_push), 3),
                        **stagnation_report,
                    })
                    _debug("commercial_finish_stagnation_guard_stop", **stagnation_report)
                    break

                unresolved = list(rep.get("unresolved_blocker_set", []) or [])
                if not unresolved:
                    unresolved = list(rep.get("hard_blocks", []) or rep.get("warnings", []) or [])
                local_failed_attempts.append(rep)
                planner = build_research_driven_blocker_plan(
                    before_analysis=before,
                    safe_reference_analysis=best_analysis,
                    failed_attempt_analysis=cand_analysis,
                    failed_attempt_report=rep,
                    unresolved_blockers=unresolved,
                    push_db=current_push,
                    floor_lufs=floor,
                    target_lufs=target,
                    virtual_hot_solver=virtual_hot_solver if isinstance(virtual_hot_solver, dict) else {},
                    decision=decision,
                    profile=profile,
                    previous_attempts=local_failed_attempts,
                )
                rep["research_driven_replan"] = planner
                convergence = planner.get("convergence", {}) if isinstance(planner.get("convergence", {}), dict) else {}
                if convergence.get("stop") or not planner.get("active"):
                    convergence_stops.append({
                        "start_push_db": round(float(start_push), 3),
                        "last_push_db": round(float(current_push), 3),
                        "reason": convergence.get("stop_reason") or "planner_inactive",
                        "unresolved_signature": planner.get("unresolved_signature"),
                    })
                    break
                bundle = planner.get("chosen_bundle", {}) if isinstance(planner.get("chosen_bundle", {}), dict) else {}
                next_push = float(bundle.get("next_push_db", current_push) or current_push)
                _v6314_next_cap = _v6314_repeated_crest_loss_push_cap(report, next_push, decision)
                if _v6314_next_cap.get("active"):
                    rep["v6314_next_push_cap"] = _v6314_next_cap
                    next_push = float(_v6314_next_cap.get("effective_push_db", next_push) or next_push)
                # If the bundle kept the push almost unchanged, that is fine: the
                # next iteration changes the responsibility split, not just level.
                current_push = float(np.clip(next_push, 0.0, max_push))
                resolution_plan = planner
                if current_push < 0.18:
                    convergence_stops.append({"start_push_db": round(float(start_push), 3), "reason": "next_push_below_useful_hot_threshold"})
                    break
            except Exception as exc:
                err = {"name": f"render_attempt_push_{current_push:.2f}db_r{round_index}", "attempt_type": "render_attempt", "error": str(exc)[:300]}
                report["render_attempts"].append(err)
                _debug("commercial_finish_render_attempt_failed", push_db=current_push, error=str(exc)[:300])
                break

    best_lufs = float(best_analysis.get("integrated_lufs", base_lufs) or base_lufs)
    selected = best_name != "base"
    report["selected"] = best_name
    report["selected_score"] = round(float(best_score), 3) if selected else None
    report["output"] = _analysis_summary_for_loudness(best_analysis)
    report["lufs_gain_achieved_db"] = round(float(best_lufs - base_lufs), 3)
    report["target_met"] = bool(best_lufs >= floor)
    report["target_reached"] = bool(best_lufs >= target - 0.20)
    report["safe_checkpoint_count"] = len(report.get("safe_checkpoints", []) or [])
    report["render_attempt_count"] = len(report.get("render_attempts", []) or [])
    report["convergence_stops"] = convergence_stops
    report["technical_safety_summary"] = {
        "elapsed_sec": round(float(_elapsed_sec()), 3),
        "total_steps": int(total_steps),
        "signature_history": signature_history,
        "hit_wall_time_limit": bool(_elapsed_sec() > wall_time_limit_sec),
        "hit_total_step_limit": bool(total_steps >= safety_max_steps),
        "safe_checkpoint_available": bool(report.get("safe_checkpoints")),
        "fallback_policy": "selected safe_checkpoint when available, otherwise base fallback",
    }
    report["selected_master_policy"] = "select highest-scoring measured safe checkpoint; if none exists, base remains selected"
    if report["target_met"]:
        report["reason"] = "commercial_floor_reached_after_research_driven_blocker_resolution" if selected else "commercial_floor_already_met"
    elif selected:
        report["reason"] = "best_safe_checkpoint_selected_but_commercial_floor_not_reached"
        report["retry_blocked_by"] = "research_planner_converged_before_floor"
    else:
        report["reason"] = "commercial_floor_not_reached_no_safe_checkpoint"
        blocks: list[str] = []
        for attempt in report.get("render_attempts", []):
            if isinstance(attempt, dict):
                blocks.extend(attempt.get("unresolved_blocker_set", []) or attempt.get("hard_blocks", []) or [])
        report["retry_blocked_by"] = sorted(set(blocks)) or "no_qc_safe_louder_checkpoint"
    if isinstance(report.get("v57_commercial_loudness_feasibility"), dict) and report["v57_commercial_loudness_feasibility"].get("decision") == "do_not_push_select_best_safe_candidate" and not selected:
        report["reason"] = "v57_no_safe_push_budget_select_base_or_existing_safe_candidate"
        report["retry_blocked_by"] = report["v57_commercial_loudness_feasibility"].get("reasons")
    if total_steps >= safety_max_steps:
        report["technical_safety_guard"] = {
            "hit": True,
            "max_steps": safety_max_steps,
            "policy": "not a musical retry limit; prevents infinite loops if convergence detection fails",
        }
    _v57_attach_damage_aware_selection_summary(report)
    _attach_loudness_checkpoint_diversity_report(report)
    return np.nan_to_num(best_audio, nan=0.0, posinf=0.0, neginf=0.0), report


def _codec_preview_guard(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Estimate lossy-codec risk without storing/encoding extra audio."""
    q_after = after.get("quality_indices", {}) if isinstance(after, dict) else {}
    q_before = before.get("quality_indices", {}) if isinstance(before, dict) else {}
    ms = after.get("ms", {}) if isinstance(after, dict) else {}
    tp = float(after.get("approx_true_peak_dbfs", -99) or -99)
    lufs = float(after.get("integrated_lufs", -99) or -99)
    crest = float(after.get("crest_factor_db", 99) or 99)
    high_side = float(ms.get("side_ratio_8k_14k", 0) or 0)
    low_side = float(ms.get("side_ratio_lowband", 0) or 0)
    harsh_delta = float(q_after.get("harshness_index_db", 0) or 0) - float(q_before.get("harshness_index_db", 0) or 0)
    air_delta = float(q_after.get("air_index_db", 0) or 0) - float(q_before.get("air_index_db", 0) or 0)
    warnings: list[str] = []
    if tp > -0.20:
        warnings.append("codec_headroom_tight")
    if lufs > -8.5 and crest < 7.2:
        warnings.append("codec_density_distortion_risk")
    if high_side > 0.62 and air_delta > 1.1:
        warnings.append("codec_side_air_smear_risk")
    if harsh_delta > 1.3:
        warnings.append("codec_harshness_risk")
    if low_side > 0.16:
        warnings.append("codec_low_end_phase_risk")
    score = 100.0 - 16.0 * len(warnings)
    if tp > -0.20:
        score -= min(12.0, (tp + 0.20) * 40.0)
    if crest < 7.5:
        score -= (7.5 - crest) * 4.0
    score = float(np.clip(score, 0, 100))
    return {
        "active": _env_bool("BUSY_CODEC_PREVIEW_GUARD", True),
        "method": "heuristic_no_encode",
        "score": round(score, 2),
        "true_peak_dbtp": round(tp, 3),
        "lufs": round(lufs, 3),
        "crest_db": round(crest, 3),
        "high_side_ratio": round(high_side, 5),
        "low_side_ratio": round(low_side, 5),
        "harshness_delta_db": round(harsh_delta, 3),
        "air_delta_db": round(air_delta, 3),
        "warnings": warnings if _env_bool("BUSY_CODEC_PREVIEW_GUARD", True) else [],
    }


def _playback_translation_report(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Estimate translation across phone/earbuds/car/mono playback from analysis."""
    q = after.get("quality_indices", {}) if isinstance(after, dict) else {}
    ms = after.get("ms", {}) if isinstance(after, dict) else {}
    corr = float(after.get("correlation", 1.0) or 1.0)
    crest = float(after.get("crest_factor_db", 99) or 99)
    low_side = float(ms.get("side_ratio_lowband", 0) or 0)
    high_side = float(ms.get("side_ratio_8k_14k", 0) or 0)
    mud = float(q.get("mud_index_db", 0) or 0)
    harsh = float(q.get("harshness_index_db", 0) or 0)
    bass = float(q.get("bass_index_db", 0) or 0)
    air = float(q.get("air_index_db", 0) or 0)

    phone = 86.0
    if bass > 24:
        phone -= min(10.0, (bass - 24) * 0.8)
    if mud > 10:
        phone -= min(12.0, (mud - 10) * 1.1)
    if harsh > 0:
        phone -= min(9.0, harsh * 0.8)
    if crest < 7.0:
        phone -= (7.0 - crest) * 3.0

    earbuds = 88.0
    if harsh > -1.0:
        earbuds -= min(12.0, (harsh + 1.0) * 1.2)
    if air > 0 and high_side > 0.55:
        earbuds -= min(10.0, air * 0.7)
    if crest < 7.0:
        earbuds -= 5.0

    car = 88.0
    if bass > 25:
        car -= min(10.0, (bass - 25) * 0.7)
    if mud > 9:
        car -= min(10.0, (mud - 9) * 0.8)
    if low_side > 0.14:
        car -= min(12.0, (low_side - 0.14) * 100.0)

    mono = 92.0
    if corr < 0.10:
        mono -= 22.0
    if low_side > 0.16:
        mono -= min(20.0, (low_side - 0.16) * 120.0)
    if high_side > 0.72 and corr < 0.25:
        mono -= 10.0

    scores = {
        "phone_speaker": round(float(np.clip(phone, 0, 100)), 2),
        "earbuds": round(float(np.clip(earbuds, 0, 100)), 2),
        "car_speaker": round(float(np.clip(car, 0, 100)), 2),
        "mono_speaker": round(float(np.clip(mono, 0, 100)), 2),
    }
    warnings: list[str] = []
    for k, v in scores.items():
        if v < 70:
            warnings.append(f"translation_{k}_risk")
    return {
        "active": _env_bool("BUSY_PLAYBACK_TRANSLATION_REPORT", True),
        "method": "heuristic_analysis_based",
        "scores": scores,
        "overall_score": round(float(np.mean(list(scores.values()))), 2),
        "warnings": warnings if _env_bool("BUSY_PLAYBACK_TRANSLATION_REPORT", True) else [],
        "diagnostics": {
            "correlation": round(corr, 4),
            "low_side_ratio": round(low_side, 5),
            "high_side_ratio": round(high_side, 5),
            "mud_index_db": round(mud, 3),
            "harshness_index_db": round(harsh, 3),
            "bass_index_db": round(bass, 3),
            "air_index_db": round(air, 3),
            "crest_db": round(crest, 3),
        },
    }

def finalize_limiter_master(
    pre_limiter: np.ndarray,
    sr: int,
    decision: dict[str, Any],
    before: dict[str, Any],
    pre_limiter_report: dict[str, Any] | None = None,
    mode: str = "Auto Commercial Master",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fresh-worker final limiting, true-peak normalization, dither, and final QC."""
    runtime_intensity_lock = _refresh_runtime_mastering_policy(decision)
    _debug("engine_start_fresh_limiter_worker", sr=sr, shape=getattr(pre_limiter, "shape", None), dtype=str(getattr(pre_limiter, "dtype", "")), mode=mode, runtime_intensity_lock=runtime_intensity_lock)
    os_factor, adaptive_quality_report = _effective_oversample_factor(before, decision)
    limiter_cfg = decision.get("limiter", {}) if isinstance(decision, dict) else {}
    lookahead_ms = limiter_cfg.get("lookahead_ms", 1.0)
    release_ms = limiter_cfg.get("release_ms", 160.0)
    if isinstance(lookahead_ms, list):
        lookahead_ms = float(sum(lookahead_ms) / len(lookahead_ms))
    if isinstance(release_ms, list):
        release_ms = float(sum(release_ms) / len(release_ms))

    # v63.1.6: after reconnecting a true pre-limiter checkpoint, the fallback
    # standard final must still run the normal loudness iteration before the
    # safety limiter.  Previously this checkpoint was already limited before
    # finalize_limiter_master(); after the reconnect it is intentionally not.
    standard_final_loudness_iteration: dict[str, Any] = {
        "schema_version": "busy_v6316_standard_final_loudness_iteration",
        "enabled": bool(_env_bool("BUSY_V6316_STANDARD_FINAL_LOUDNESS_ITERATION", True)),
        "active": False,
        "reason": "not_run",
        "policy": "Run the normal loudness iteration in finalize_limiter_master so the standard fallback is a real mastered reference even when commercial candidate QC rejects all alternatives.",
    }
    standard_final_source = pre_limiter
    if standard_final_loudness_iteration.get("enabled"):
        try:
            _proc0 = (pre_limiter_report or {}).get("processing_report", {}) if isinstance(pre_limiter_report, dict) else {}
            _iter0 = _proc0.get("loudness_iteration", {}) if isinstance(_proc0, dict) and isinstance(_proc0.get("loudness_iteration"), dict) else {}
            _target_lufs, _ceil_db = _target_from_decision(decision, mode)
            if _iter0.get("target_lufs_preview") is not None:
                _target_lufs = float(_iter0.get("target_lufs_preview"))
            if _iter0.get("ceiling_db_preview") is not None:
                _ceil_db = float(_iter0.get("ceiling_db_preview"))
            _debug("engine_start_v6316_standard_final_loudness_iteration", target_lufs=round(float(_target_lufs), 3), ceiling_db=round(float(_ceil_db), 3))
            standard_final_source, _std_iter_report = _loudness_iteration(pre_limiter, sr, target_lufs=float(_target_lufs), ceiling_db=float(_ceil_db), limiter=limiter_cfg, max_iters=7)
            standard_final_loudness_iteration.update({
                "active": True,
                "reason": "completed",
                "target_lufs": round(float(_target_lufs), 3),
                "ceiling_db": round(float(_ceil_db), 3),
                "iteration_report": _std_iter_report,
            })
            _std_iter_analysis = analyze_audio_fast_qc(standard_final_source, sr, true_peak_oversample=_qc_oversample_factor())
            standard_final_loudness_iteration["output_analysis"] = _analysis_summary_for_loudness(_std_iter_analysis)
            _debug("engine_done_v6316_standard_final_loudness_iteration", lufs=_std_iter_analysis.get("integrated_lufs"), true_peak=_std_iter_analysis.get("approx_true_peak_dbfs"))
        except Exception as exc:
            standard_final_loudness_iteration.update({"active": False, "reason": "exception_fallback_to_raw_pre_limiter", "error": str(exc)[:260]})
            standard_final_source = pre_limiter
            _debug("engine_error_v6316_standard_final_loudness_iteration", error=str(exc)[:260])

    _debug("engine_start_oversampled_final_limiter", oversample=os_factor, ceiling_db=FINAL_TRUE_PEAK_CEILING_DB, lookahead_ms=float(lookahead_ms), release_ms=float(release_ms))
    final = oversampled_limit_chunked(
        standard_final_source,
        sr,
        ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
        oversample=os_factor,
        lookahead_ms=float(lookahead_ms),
        release_ms=float(release_ms),
    )
    _debug("engine_done_oversampled_final_limiter", output_shape=getattr(final, "shape", None))

    _debug("engine_start_true_peak_ceiling_attenuation", oversample=os_factor, target_db=FINAL_TRUE_PEAK_CEILING_DB)
    final, final_true_peak_gain_db, true_peak_ceiling_attenuation = _attenuate_to_true_peak_ceiling(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=os_factor)
    _debug("engine_done_true_peak_ceiling_attenuation", gain_db=round(final_true_peak_gain_db, 3), active=true_peak_ceiling_attenuation.get("active"), reason=true_peak_ceiling_attenuation.get("reason"), pre_true_peak_dbtp=true_peak_ceiling_attenuation.get("pre_true_peak_dbtp"))

    final, optimizer_report = _multi_pass_auto_optimizer(final, sr, before, decision, mode)
    if optimizer_report.get("active"):
        _debug("engine_done_multi_pass_optimizer", selected=optimizer_report.get("selected"), selected_score=optimizer_report.get("selected_score"), base_score=optimizer_report.get("base_score"))

    pre_limiter_report = pre_limiter_report or {}
    experimental_residue_light_control = pre_limiter_report.get("residue_pre_clean_stage") or {
        "active": False,
        "reason": "pre_clean_report_missing_or_not_applied",
        "stage": "pre_clean_before_premaster",
        "policy": "No post-master residue trimming; direct cleanup, if needed, must happen once before premaster.",
    }
    _debug(
        "engine_skip_post_residue_light_control",
        reason="residue_cleanup_is_pre_master_once",
        pre_clean_active=experimental_residue_light_control.get("active"),
        action_count=len(experimental_residue_light_control.get("actions") or []),
    )

    # v8.5.3.19: commercial finish must evaluate and apply the virtual strategy
    # against the pre-limiter signal, not against an already limited master.
    # Running the hot solver after the first safety limiter made +5~+11 dB
    # candidates collapse back into the limiter and report almost no LUFS gain.
    # v63.1.9: single-stage true pre-limiter handoff is not just telemetry.
    # If the checkpoint render succeeded, commercial finish must use it rather
    # than the already-limited standard final.
    if "_handoff_audio" in locals() and isinstance(v6316_true_pre_limiter_handoff, dict) and v6316_true_pre_limiter_handoff.get("active"):
        commercial_source_for_finish = _handoff_audio
        commercial_source_label = "true_pre_limiter_handoff"
    elif "pre_limiter" in locals():
        commercial_source_for_finish = pre_limiter
        commercial_source_label = "pre_limiter"
    else:
        commercial_source_for_finish = final
        commercial_source_label = "standard_final"
    _commercial_finish_started_at = time.monotonic()
    _debug(
        "engine_start_commercial_loudness_finish",
        source=commercial_source_label,
        shape=getattr(commercial_source_for_finish, "shape", None),
        sr=sr,
    )
    commercial_candidate, commercial_finish_report = _commercial_loudness_finish_pass(commercial_source_for_finish, sr, before, decision, mode)
    try:
        commercial_finish_report["single_stage_commercial_finish_source"] = commercial_source_label
        commercial_finish_report["single_stage_true_pre_limiter_handoff_connected"] = bool(commercial_source_label in {"true_pre_limiter_handoff", "pre_limiter"})
        commercial_finish_report["v6319_bamix_handoff_mode"] = _bamix_v6319_handoff_mode(decision)
    except Exception:
        pass
    try:
        commercial_finish_report["runtime_elapsed_sec"] = round(max(0.0, time.monotonic() - _commercial_finish_started_at), 3)
    except Exception:
        pass
    _debug(
        "engine_done_commercial_loudness_finish_pass",
        selected=commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None,
        reason=commercial_finish_report.get("reason") if isinstance(commercial_finish_report, dict) else None,
        elapsed_sec=commercial_finish_report.get("runtime_elapsed_sec") if isinstance(commercial_finish_report, dict) else None,
        virtual_runtime=(commercial_finish_report.get("virtual_hot_commercial_solver") or {}).get("runtime_proxy_guard") if isinstance(commercial_finish_report, dict) and isinstance(commercial_finish_report.get("virtual_hot_commercial_solver"), dict) else None,
    )
    commercial_selected = bool(commercial_finish_report.get("selected") and commercial_finish_report.get("selected") != "base")
    commercial_finish_report["strategy_application_stage"] = "pre_limiter_before_final_safety_limiter"
    # v8.5.3.20: never replace the already rendered standard final with a
    # commercial candidate that is quieter.  The finish pass is evaluated against
    # the pre-limiter source, so its internal base can be much quieter than the
    # standard final.  This guard prevents a "selected" pre-limiter candidate from
    # making the delivered master smaller.
    standard_final_audio_for_fallback = final
    standard_final_reference = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    commercial_finish_report["standard_final_reference_analysis"] = _analysis_summary_for_loudness(standard_final_reference)
    if commercial_selected:
        _selected_fg_limiter_cfg = {**(limiter_cfg if isinstance(limiter_cfg, dict) else {}), "candidate_analysis": commercial_finish_report.get("output") if isinstance(commercial_finish_report, dict) else None, "candidate_label": commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None}
        commercial_candidate, _final_grade_report = _final_grade_selected_candidate_render(
            commercial_candidate,
            sr,
            before,
            decision,
            limiter_cfg=_selected_fg_limiter_cfg,
        )
        commercial_finish_report["selected_candidate_final_grade_render"] = _final_grade_report
        if isinstance(_final_grade_report, dict):
            _v56_pref_output = _final_grade_report.get("post_analysis_full") if isinstance(_final_grade_report.get("post_analysis_full"), dict) and _final_grade_report.get("post_analysis_full") else _final_grade_report.get("post_analysis")
            if isinstance(_v56_pref_output, dict):
                commercial_finish_report["output"] = _v56_pref_output
                commercial_finish_report["output_analysis_mode"] = "full_analyze_audio" if _final_grade_report.get("post_analysis_full") else "fast_qc"
            if isinstance(_final_grade_report.get("buffer_identity"), dict):
                commercial_finish_report["selected_candidate_export_identity"] = _final_grade_report.get("buffer_identity")
            commercial_finish_report["v56_candidate_handoff_export_identity_lock"] = {
                "schema_version": "busy_v56_candidate_handoff_export_identity_lock",
                "active": True,
                "selected_candidate": commercial_finish_report.get("selected"),
                "expected_analysis_mode": commercial_finish_report.get("output_analysis_mode"),
                "selected_candidate_buffer_sha256_16": (_final_grade_report.get("buffer_identity") or {}).get("sha256_16") if isinstance(_final_grade_report.get("buffer_identity"), dict) else None,
                "policy": "selected-candidate output uses full export analysis when available so export lock compares the same analyzer family.",
            }
            # v8.5.3.57.1: the v57 damage gate must be applied to the
            # final-grade selected-candidate render as well, not only to the
            # earlier candidate checkpoint. v56 can legitimately make the
            # final-grade buffer louder than the checkpoint, so LRA/transient
            # damage must be checked on the actual buffer that would replace the
            # standard final.
            try:
                _v57_final_grade_analysis = _v56_pref_output if isinstance(_v56_pref_output, dict) else {}
                _v57_final_grade_warnings = _safety_risk(before, _v57_final_grade_analysis, decision) if _v57_final_grade_analysis else []
                _v57_final_grade_profile = _commercial_loudness_finish_profile(decision, mode)
                _v57_final_grade_gate = _v57_damage_aware_candidate_gate(
                    before,
                    _v57_final_grade_analysis,
                    standard_final_reference,
                    decision,
                    _v57_final_grade_profile,
                    _v57_final_grade_warnings,
                    float(commercial_finish_report.get("commercial_floor_lufs", commercial_finish_report.get("target_lufs", -9.0)) or -9.0),
                    float(commercial_finish_report.get("target_lufs", -8.0) or -8.0),
                ) if _v57_final_grade_analysis else {"enabled": _env_bool("BUSY_V57_DAMAGE_AWARE_SAFE_SELECTION", True), "active": False, "reason": "missing_final_grade_analysis"}
                commercial_finish_report["selected_candidate_final_grade_v57_warnings"] = _v57_final_grade_warnings
                commercial_finish_report["selected_candidate_final_grade_v57_damage_gate"] = _v57_final_grade_gate
                if isinstance(_final_grade_report, dict):
                    _final_grade_report["v57_damage_aware_candidate_gate"] = _v57_final_grade_gate
                    _final_grade_report["v57_warnings"] = _v57_final_grade_warnings
                if _v57_final_grade_gate.get("reject"):
                    commercial_selected = False
                    commercial_finish_report["candidate_rejected_after_v57_final_grade_damage_gate"] = True
                    commercial_finish_report["candidate_rejection_reason"] = "v57_final_grade_damage_gate"
                    commercial_finish_report["candidate_rejection_problem_set"] = _v57_final_grade_gate.get("blockers", [])
                    commercial_finish_report["fallback_final_source"] = "standard_final_after_v57_damage_gate"
                    commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
                    commercial_finish_report["output"] = commercial_finish_report.get("standard_final_reference_analysis") or commercial_finish_report.get("output")
                    commercial_finish_report["output_analysis_mode"] = "standard_final_after_v57_damage_gate"
                _v57_attach_damage_aware_selection_summary(commercial_finish_report)
            except Exception as exc:
                commercial_finish_report["selected_candidate_final_grade_v57_damage_gate"] = {
                    "schema_version": "busy_v57_final_grade_damage_gate_v8_5_3_57_2",
                    "active": False,
                    "reason": "exception",
                    "error": str(exc)[:240],
                }
    if commercial_selected:
        _v6314_final_choice = _v6314_candidate_dynamic_governance(standard_final_reference, commercial_finish_report.get("output") if isinstance(commercial_finish_report.get("output"), dict) else {}, decision, float(commercial_finish_report.get("commercial_floor_lufs", -99.0) or -99.0), candidate_label=str(commercial_finish_report.get("selected") or "selected_candidate"))
        commercial_finish_report["v6314_cleaner_quieter_final_choice"] = _v6314_final_choice
        if _v6314_final_choice.get("reject_for_cleaner_quieter"):
            commercial_selected = False
            commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
            commercial_finish_report["candidate_rejected_after_v6314_cleaner_quieter_guard"] = True
            commercial_finish_report["candidate_rejection_reason"] = "v6314_cleaner_quieter_standard_final_preserved_after_target_met"
            commercial_finish_report["fallback_final_source"] = "standard_final_limiter_output"
            commercial_finish_report["selected"] = "standard_final_limiter_output"
            commercial_finish_report["selected_final_source"] = "standard_final_limiter_output"
            commercial_finish_report["delivered_final_is_standard_after_v6314_cleaner_guard"] = True
            commercial_finish_report["output"] = _analysis_summary_for_loudness(standard_final_reference)
            commercial_finish_report["delivered_final_output"] = _analysis_summary_for_loudness(standard_final_reference)
            try:
                _v6314_std_floor = float(commercial_finish_report.get("commercial_floor_lufs", -99.0) or -99.0)
                _v6314_std_lufs = float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)
                commercial_finish_report["target_met"] = bool(_v6314_std_lufs >= _v6314_std_floor)
                commercial_finish_report["target_reached"] = bool(_v6314_std_lufs >= float(commercial_finish_report.get("target_lufs", _v6314_std_floor) or _v6314_std_floor) - 0.20)
                commercial_finish_report["reason"] = "v6314_cleaner_quieter_standard_final_preserved_target_met" if commercial_finish_report["target_met"] else "v6314_cleaner_quieter_standard_final_preserved_below_floor"
            except Exception:
                commercial_finish_report["target_met"] = commercial_finish_report.get("target_met")
            final = standard_final_audio_for_fallback

    try:
        std_lufs = float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)
        cand_lufs = float((commercial_finish_report.get("output") or {}).get("integrated_lufs", -99.0) or -99.0)
    except Exception:
        std_lufs, cand_lufs = -99.0, -99.0
    if commercial_selected and cand_lufs < std_lufs + 0.12 and not bool(commercial_finish_report.get("target_met")):
        commercial_finish_report["candidate_rejected_against_standard_final"] = True
        _std_floor = float(commercial_finish_report.get("commercial_floor_lufs", -99.0) or -99.0)
        _std_target = float(commercial_finish_report.get("target_lufs", _std_floor) or _std_floor)
        _std_target_met = bool(std_lufs >= _std_target - 0.20)
        _std_floor_met = bool(std_lufs >= _std_floor - 0.03)
        commercial_finish_report["candidate_rejection_reason"] = "candidate_not_louder_than_standard_final_standard_preserved_target_met" if _std_floor_met else "candidate_not_louder_than_standard_final"
        commercial_finish_report["candidate_vs_standard_lufs_delta_db"] = round(float(cand_lufs - std_lufs), 3)
        commercial_finish_report["standard_final_target_met"] = _std_target_met
        commercial_finish_report["standard_final_floor_met"] = _std_floor_met
        if _std_floor_met:
            commercial_finish_report["target_met"] = True
            commercial_finish_report["target_reached"] = _std_target_met
            commercial_finish_report["reason"] = "standard_final_preserved_candidate_not_louder_target_met"
            commercial_finish_report["output"] = _analysis_summary_for_loudness(standard_final_reference)
            commercial_finish_report["delivered_final_output"] = _analysis_summary_for_loudness(standard_final_reference)
        commercial_selected = False
    # v8.5.3.21: before replacing the standard final, audit the candidate as
    # an actual final output.  A candidate that is louder but introduces a new
    # critical problem set is not accepted-with-warnings; it is rejected and the
    # standard final remains the delivered master.
    finish_profile_for_guard = _commercial_loudness_finish_profile(decision, mode)
    if commercial_selected:
        _debug("commercial_finish_final_replacement_guard_start", selected=commercial_finish_report.get("selected"), cached_candidate_analysis=bool(isinstance(commercial_finish_report.get("output"), dict) and commercial_finish_report.get("output")))
        immediate_candidate_analysis = commercial_finish_report.get("output") if isinstance(commercial_finish_report.get("output"), dict) and commercial_finish_report.get("output") else analyze_audio_fast_qc(commercial_candidate, sr, true_peak_oversample=_qc_oversample_factor())
        standard_problem = _candidate_problem_set(before, standard_final_reference, decision, finish_profile_for_guard, _safety_risk(before, standard_final_reference, decision), base_problem_set=[])
        candidate_problem = _candidate_problem_set(before, immediate_candidate_analysis, decision, finish_profile_for_guard, _safety_risk(before, immediate_candidate_analysis, decision), base_problem_set=list(standard_problem.get("problem_set", []) or []))
        v6299_arbiter = _v6299_final_perceptual_ab_arbiter(before, standard_final_reference, immediate_candidate_analysis, decision, candidate_problem)
        commercial_finish_report["v62_9_9_final_perceptual_ab_arbiter"] = v6299_arbiter
        commercial_finish_report["final_replacement_guard"] = {
            "schema_version": "busy_selected_candidate_final_replacement_guard_v8_5_3_21_plus_v62_9_9_perceptual_arbiter",
            "standard_problem_set": standard_problem,
            "candidate_problem_set": candidate_problem,
            "v62_9_9_final_perceptual_ab_arbiter": v6299_arbiter,
            "accepted": bool((not candidate_problem.get("critical_new_vs_base")) and not bool(v6299_arbiter.get("reject"))),
            "policy": "selected candidate may replace standard final only if it introduces no new critical problem set and passes the v62.9.9 perceptual/LRA/crest arbiter",
        }
        if candidate_problem.get("critical_new_vs_base") or bool(v6299_arbiter.get("reject")):
            commercial_selected = False
            commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
            commercial_finish_report["candidate_rejected_after_final_guard"] = True
            commercial_finish_report["candidate_rejection_reason"] = "v62_9_9_final_perceptual_ab_arbiter" if bool(v6299_arbiter.get("reject")) and not candidate_problem.get("critical_new_vs_base") else "new_unresolved_problem_set_after_final_guard"
            commercial_finish_report["candidate_rejection_problem_set"] = candidate_problem.get("critical_new_vs_base") or v6299_arbiter.get("reasons")
            final = standard_final_audio_for_fallback
            commercial_finish_report = _v6299_normalize_commercial_finish_after_guard(commercial_finish_report, standard_final_reference)
        else:
            commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = True
            final = commercial_candidate
    else:
        commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
        commercial_finish_report.setdefault("fallback_final_source", "standard_final_limiter_output")
    _debug(
        "engine_done_commercial_loudness_finish",
        active=commercial_finish_report.get("active"),
        target_met=commercial_finish_report.get("target_met"),
        selected=commercial_finish_report.get("selected"),
        reason=commercial_finish_report.get("reason"),
        output=(commercial_finish_report.get("output") or {}).get("integrated_lufs"),
        strategy_application_stage=commercial_finish_report.get("strategy_application_stage"),
    )
    try:
        final = np.nan_to_num(np.asarray(final, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        final = np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0)
    _cached_final_analysis_for_runtime = commercial_finish_report.get("output") if isinstance(commercial_finish_report, dict) and isinstance(commercial_finish_report.get("output"), dict) else {}

    # v8.5.3.34: final safety ceiling enforcement before selected-master lock.
    # Optimizer/candidate replacement can slightly move true peak after the first
    # limiter.  The final selected chain must still obey the locked intensity TP
    # ceiling before dither/export.
    final_safety_ceiling_enforcement = {
        "schema_version": "busy_final_safety_ceiling_enforcement_v8_5_3_34",
        "active": False,
        "target_dbtp": round(float(FINAL_TRUE_PEAK_CEILING_DB), 3),
        "policy": "hard TP ceiling is enforced before selected-master lock; no post-lock loudness/gain pass is allowed",
    }
    try:
        _pre_final_tp_analysis = _cached_final_analysis_for_runtime if isinstance(_cached_final_analysis_for_runtime, dict) and _cached_final_analysis_for_runtime else analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
        _pre_final_tp = float(_pre_final_tp_analysis.get("approx_true_peak_dbfs", _pre_final_tp_analysis.get("true_peak_dbfs", -99.0)) or -99.0)
        final_safety_ceiling_enforcement["pre_enforcement_true_peak_dbtp"] = round(_pre_final_tp, 3)
        if _pre_final_tp > float(FINAL_TRUE_PEAK_CEILING_DB) + 0.03:
            final, _ceiling_gain_db = _normalize_to_true_peak(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=_qc_oversample_factor())
            final_safety_ceiling_enforcement.update({
                "active": True,
                "gain_db": round(float(_ceiling_gain_db), 3),
            })
            _post_final_tp_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
            final_safety_ceiling_enforcement["post_enforcement_true_peak_dbtp"] = round(float(_post_final_tp_analysis.get("approx_true_peak_dbfs", -99.0) or -99.0), 3)
            _debug("engine_done_final_safety_ceiling_enforcement", **final_safety_ceiling_enforcement)
        else:
            final_safety_ceiling_enforcement["reason"] = "already_within_locked_true_peak_ceiling"
    except Exception as exc:
        final_safety_ceiling_enforcement.update({"active": False, "reason": "exception", "error": str(exc)[:300]})
        _debug("engine_error_final_safety_ceiling_enforcement", error=str(exc)[:300])

    _pre_lra_recovery_analysis = _cached_final_analysis_for_runtime if isinstance(_cached_final_analysis_for_runtime, dict) and _cached_final_analysis_for_runtime else analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    final, lra_recovery_stage = _maybe_apply_lra_recovery_stage(final, sr, before, _pre_lra_recovery_analysis, decision, mode, commercial_finish_report)
    _debug("engine_done_lra_recovery_stage", active=lra_recovery_stage.get("active"), applied=lra_recovery_stage.get("applied"), reason=lra_recovery_stage.get("reason"), candidate=lra_recovery_stage.get("candidate", {}))

    # v8.5.3.60.1: run the deterministic multistage limiter in the fresh
    # two-stage/finalize_limiter_master path as well. v60 originally wired DML
    # into the single-stage path only, so production two-stage jobs showed the
    # v60 report schema while the DML report stayed empty. This block must run
    # before dither and final analysis so even rejected/shadow DML candidates
    # leave explicit telemetry.
    if _env_bool("BUSY_V60_DML_FULL_ANALYSIS", False):
        try:
            _v60_standard_analysis = analyze_audio(final, sr)
        except Exception:
            _v60_standard_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    else:
        _v60_standard_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    # v63.1.6.2: run DML as a final-stage multistage architecture against
    # the current delivered final reference, not the raw pre-limiter checkpoint.
    _v60_standard_analysis = _v6362_ensure_lra_for_dml(_v60_standard_analysis, final, sr, source="standard_final_reference_before_dml")
    _v60_source = final
    _debug("engine_start_v60_deterministic_multistage_limiter", source="current_final_reference", source_shape=getattr(_v60_source, "shape", None), final_shape=getattr(final, "shape", None), cached_standard_analysis=bool(isinstance(_cached_final_analysis_for_runtime, dict) and _cached_final_analysis_for_runtime), standard_lra=_analysis_lra_lu(_v60_standard_analysis), dml_engine_version=V6362_DML_ENGINE_VERSION)
    _v60_candidate_audio, v60_deterministic_multistage_limiter = _v60_deterministic_multistage_limiter(
        _v60_source,
        sr,
        before,
        decision,
        _v60_standard_analysis,
        os_factor=os_factor,
        finish_report=commercial_finish_report,
    )
    if isinstance(v60_deterministic_multistage_limiter, dict):
        v60_deterministic_multistage_limiter["v63162_source_role"] = "current_final_reference_after_commercial_finish_and_final_safety"
        v60_deterministic_multistage_limiter["engine_version"] = V6362_DML_ENGINE_VERSION
        _v6362_contract = v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else (v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {}
        if isinstance(_v6362_contract, dict) and _v6362_contract:
            commercial_finish_report["v6362_dml_ownership_contract"] = _v6362_contract
    if isinstance(v60_deterministic_multistage_limiter, dict) and v60_deterministic_multistage_limiter.get("applied"):
        final = _v60_candidate_audio
        commercial_finish_report["selected_final_source"] = "v60_deterministic_multistage_limiter"
        commercial_finish_report["fallback_final_source"] = "v60_deterministic_multistage_limiter"
        commercial_finish_report["v60_dml_replaced_standard_final"] = True
    elif isinstance(v60_deterministic_multistage_limiter, dict):
        commercial_finish_report["v60_dml_preserved_existing_final"] = True
    if isinstance(commercial_finish_report, dict):
        commercial_finish_report["v60_deterministic_multistage_limiter"] = v60_deterministic_multistage_limiter
    _debug(
        "engine_done_v60_deterministic_multistage_limiter",
        active=(v60_deterministic_multistage_limiter or {}).get("active") if isinstance(v60_deterministic_multistage_limiter, dict) else None,
        accepted=(v60_deterministic_multistage_limiter or {}).get("accepted") if isinstance(v60_deterministic_multistage_limiter, dict) else None,
        applied=(v60_deterministic_multistage_limiter or {}).get("applied") if isinstance(v60_deterministic_multistage_limiter, dict) else None,
        reason=(v60_deterministic_multistage_limiter or {}).get("reason") if isinstance(v60_deterministic_multistage_limiter, dict) else None,
    )

    _pre_v6361_dtp_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    # Cache now matches the current post-DML final. If v63.6.1 applies audio,
    # the cache is cleared below so final analysis is recomputed.
    _cached_final_analysis_for_runtime = _pre_v6361_dtp_analysis
    final, v6361_drum_transient_punch = apply_drum_transient_punch_complete(
        final,
        sr,
        input_analysis=before if isinstance(before, dict) else {},
        current_analysis=_pre_v6361_dtp_analysis,
        decision=decision if isinstance(decision, dict) else {},
        mode=mode,
        true_peak_ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
        oversample=_qc_oversample_factor(),
        context={
            "stage": "two_stage_finalize_after_dml_before_dither",
            "commercial_finish_report": commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
            "v6362_dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "v60_deterministic_multistage_limiter": v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {},
        },
        analyzer=lambda audio, rate: analyze_audio_fast_qc(audio, rate, true_peak_oversample=_qc_oversample_factor()),
    )
    if isinstance(v6361_drum_transient_punch, dict) and v6361_drum_transient_punch.get("applied"):
        _cached_final_analysis_for_runtime = {}
    if isinstance(commercial_finish_report, dict):
        commercial_finish_report["v6361_drum_transient_punch"] = v6361_drum_transient_punch
    _debug("engine_done_v6361_drum_transient_punch", active=v6361_drum_transient_punch.get("active"), applied=v6361_drum_transient_punch.get("applied"), reason=v6361_drum_transient_punch.get("reason"), selected_score=v6361_drum_transient_punch.get("selected_score"))

    # v64.0.4: Commercial/Maximum density push foundation for the two-stage
    # finalize path.  Mirrors the single-stage insertion so Cloud Run exports
    # receive the actual rendered -0.1 dBTP density candidate instead of only
    # telemetry or a floor-met clean fallback.
    v6404_commercial_maximum_density = {
        "schema_version": "busy_commercial_maximum_density_v8_5_3_64_0_4",
        "engine_version": V6404_COMMERCIAL_DENSITY_ENGINE_VERSION,
        "enabled": False,
        "active": False,
        "applied": False,
        "reason": "not_run",
    }
    try:
        _pre_v6404_density_analysis = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
        final, v6404_commercial_maximum_density = apply_commercial_maximum_density_push(
            final,
            sr,
            before_analysis=before if isinstance(before, dict) else {},
            current_analysis=_pre_v6404_density_analysis,
            decision=decision if isinstance(decision, dict) else {},
            mode=mode,
            commercial_finish_report=commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
            true_peak_ceiling_db=-0.1 if str(mode or "").lower().find("safe") < 0 else FINAL_TRUE_PEAK_CEILING_DB,
            oversample=_qc_oversample_factor(),
            analyzer=lambda audio, rate: analyze_audio_fast_qc(audio, rate, true_peak_oversample=_qc_oversample_factor()),
        )
        if isinstance(v6404_commercial_maximum_density, dict) and v6404_commercial_maximum_density.get("applied"):
            _cached_final_analysis_for_runtime = v6404_commercial_maximum_density.get("selected_analysis") if isinstance(v6404_commercial_maximum_density.get("selected_analysis"), dict) else {}
            if isinstance(commercial_finish_report, dict):
                commercial_finish_report["selected_final_source"] = "v6404_commercial_maximum_density"
                commercial_finish_report["fallback_final_source"] = "v6404_commercial_maximum_density"
        elif isinstance(v6404_commercial_maximum_density, dict):
            _cached_final_analysis_for_runtime = _pre_v6404_density_analysis
        if isinstance(commercial_finish_report, dict):
            commercial_finish_report["v6404_commercial_maximum_density"] = {k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"}
        if isinstance(decision, dict):
            decision["v6404_commercial_maximum_density"] = {k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"}
        _debug(
            "engine_done_v6404_commercial_maximum_density",
            active=v6404_commercial_maximum_density.get("active") if isinstance(v6404_commercial_maximum_density, dict) else None,
            applied=v6404_commercial_maximum_density.get("applied") if isinstance(v6404_commercial_maximum_density, dict) else None,
            mode_policy=v6404_commercial_maximum_density.get("mode_policy") if isinstance(v6404_commercial_maximum_density, dict) else None,
            selected=v6404_commercial_maximum_density.get("selected_candidate_id") if isinstance(v6404_commercial_maximum_density, dict) else None,
            final_tp=v6404_commercial_maximum_density.get("final_true_peak_dbtp") if isinstance(v6404_commercial_maximum_density, dict) else None,
            under_pushed=v6404_commercial_maximum_density.get("under_pushed") if isinstance(v6404_commercial_maximum_density, dict) else None,
        )
    except Exception as exc:
        v6404_commercial_maximum_density = {
            "schema_version": "busy_commercial_maximum_density_v8_5_3_64_0_4",
            "engine_version": V6404_COMMERCIAL_DENSITY_ENGINE_VERSION,
            "enabled": True,
            "active": False,
            "applied": False,
            "reason": "exception_preserve_existing_final",
            "error": str(exc)[:300],
        }
        if isinstance(commercial_finish_report, dict):
            commercial_finish_report["v6404_commercial_maximum_density"] = v6404_commercial_maximum_density
        _debug("engine_error_v6404_commercial_maximum_density", error=str(exc)[:300])

    final, dither_report = _apply_output_dither(final, bit_depth=24)
    if dither_report.get("active"):
        _debug("engine_done_dither_noise_shaping", **dither_report)

    _debug("engine_start_final_analysis", runtime_safe=not _env_bool("BUSY_FINAL_ANALYSIS_FULL", False))
    _final_analysis_started_at = time.monotonic()
    final_analysis = _runtime_safe_final_analysis(final, sr, _cached_final_analysis_for_runtime if "_cached_final_analysis_for_runtime" in locals() else None)
    _debug("engine_done_final_analysis", elapsed_sec=round(max(0.0, time.monotonic() - _final_analysis_started_at), 3), schema=final_analysis.get("schema_version"), lufs=final_analysis.get("integrated_lufs"), true_peak=final_analysis.get("approx_true_peak_dbfs"))

    final_export_lock = {
        "schema_version": "busy_final_export_lock_v8_5_3_63_1_3_3_export_lock_not_applicable_fix",
        "enabled": _env_bool("BUSY_V48_2_2_FINAL_EXPORT_LOCK", True),
        "active": False,
        "accepted": True,
        "comparison_analyzer": "not_applicable_standard_final_or_no_selected_candidate",
        "mismatch_reasons": [],
        "buffer_identity_match": None,
        "identity_stage": "engine_return_buffer_consistency_guard",
        "policy": "selected commercial candidate full export analysis and buffer identity must match the final returned buffer before export; when no candidate replaced the standard final, the lock is not applicable and must not be reported as rejected.",
    }
    try:
        _selected_render_report = commercial_finish_report.get("selected_candidate_final_grade_render", {}) if isinstance(commercial_finish_report, dict) else {}
        _expected_selected, _expected_analysis_mode = _v56_selected_export_expected_analysis(_selected_render_report)
        _candidate_replaced = bool(commercial_finish_report.get("selected_candidate_replaced_standard_final_limiter")) if isinstance(commercial_finish_report, dict) else False
        if final_export_lock["enabled"] and _candidate_replaced and isinstance(_expected_selected, dict) and _expected_selected:
            _delta = _v48_2_2_export_lock_delta(_expected_selected, final_analysis)
            _mismatch = _v48_2_2_export_lock_mismatch(_delta)
            if final_analysis.get("final_analysis_runtime_safe") and _env_bool("BUSY_EXPORT_LOCK_RUNTIME_SAFE_NO_ROLLBACK", True):
                final_export_lock["runtime_safe_analysis_rollback_suppressed"] = bool(_mismatch)
                final_export_lock["runtime_safe_original_mismatch_reasons"] = list(_mismatch or [])
                _mismatch = []
            _returned_identity = _v56_audio_buffer_identity(
                final,
                sr,
                final_analysis,
                stage="engine_return_buffer_before_export_lock_decision",
                source="selected_candidate_current_final" if _candidate_replaced else "standard_final",
                candidate_label=str(commercial_finish_report.get("selected") or "") if isinstance(commercial_finish_report, dict) else None,
            )
            _selected_identity = _selected_render_report.get("buffer_identity") if isinstance(_selected_render_report, dict) and isinstance(_selected_render_report.get("buffer_identity"), dict) else {}
            _identity_match = (bool(_selected_identity and _returned_identity and _selected_identity.get("sha256") == _returned_identity.get("sha256")) if _selected_identity else None)
            final_export_lock.update({
                "schema_version": "busy_final_export_lock_v8_5_3_63_1_3_3_export_lock_not_applicable_fix",
                "active": True,
                "candidate_id": commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None,
                "delta": _delta,
                "mismatch_reasons": _mismatch,
                "accepted": not bool(_mismatch),
                "comparison_analyzer": _expected_analysis_mode,
                "selected_candidate_buffer_identity": _selected_identity,
                "returned_buffer_identity": _returned_identity,
                "buffer_identity_match": _identity_match,
                "false_mismatch_guard": {
                    "schema_version": "busy_v56_export_lock_false_mismatch_guard",
                    "active": _expected_analysis_mode == "full_analyze_audio",
                    "reason": "selected candidate expected analysis uses full analyze_audio to match final export analysis" if _expected_analysis_mode == "full_analyze_audio" else "legacy fast_qc expected analysis fallback",
                },
            })
            if isinstance(commercial_finish_report, dict):
                commercial_finish_report["v56_export_identity_lock"] = {
                    "schema_version": "busy_v56_candidate_handoff_export_identity_lock",
                    "active": True,
                    "candidate_id": commercial_finish_report.get("selected"),
                    "comparison_analyzer": _expected_analysis_mode,
                    "accepted": not bool(_mismatch),
                    "mismatch_reasons": _mismatch,
                    "selected_sha256_16": _selected_identity.get("sha256_16") if isinstance(_selected_identity, dict) else None,
                    "returned_sha256_16": _returned_identity.get("sha256_16") if isinstance(_returned_identity, dict) else None,
                    "buffer_identity_match": _identity_match,
                    "policy": "same selected-candidate lineage must use comparable analysis and tracked buffer identity before export.",
                }
            if _mismatch and _env_bool("BUSY_V48_2_2_EXPORT_LOCK_ROLLBACK", True):
                final_export_lock["rollback_to_standard_final"] = True
                final_export_lock["rollback_reason"] = "selected_candidate_output_analysis_mismatch"
                try:
                    final = np.nan_to_num(np.asarray(standard_final_audio_for_fallback, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                    final, _lock_tp_gain, _lock_tp_report = _attenuate_to_true_peak_ceiling(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=_qc_oversample_factor())
                    final_export_lock["fallback_true_peak_normalize_gain_db"] = round(float(_lock_tp_gain), 3)
                    final_export_lock["fallback_true_peak_ceiling_attenuation"] = _lock_tp_report
                    final, _lock_dither_report = _apply_output_dither(final, bit_depth=24)
                    final_export_lock["fallback_dither_report"] = _lock_dither_report
                    final_analysis = _runtime_safe_final_analysis(final, sr, None)
                    final_export_lock["fallback_output_analysis"] = _analysis_summary_for_loudness(final_analysis)
                    if isinstance(commercial_finish_report, dict):
                        commercial_finish_report["final_export_lock_rejected_selected_candidate"] = True
                        commercial_finish_report["final_export_lock_rejection_reasons"] = _mismatch
                        commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = False
                        commercial_finish_report["fallback_final_source"] = "standard_final_after_export_lock"
                except Exception as exc:
                    final_export_lock["rollback_error"] = str(exc)[:300]
        else:
            final_export_lock.update({
                "active": False,
                "accepted": True,
                "reason": "no_selected_commercial_candidate_or_expected_analysis_missing",
                "comparison_analyzer": "not_applicable_standard_final_or_no_selected_candidate",
                "mismatch_reasons": [],
                "buffer_identity_match": None,
                "not_applicable_guard": {
                    "schema_version": "busy_export_lock_not_applicable_guard_v8_5_3_63_1_3_3",
                    "candidate_replaced_standard_final": bool(_candidate_replaced),
                    "expected_analysis_available": bool(isinstance(_expected_selected, dict) and _expected_selected),
                    "policy": "An inactive export lock is not a rejection; standard-final or no-expected-analysis paths are marked accepted-but-not-applicable for debug/Supabase clarity.",
                },
            })
    except Exception as exc:
        final_export_lock.update({"active": False, "reason": "exception", "error": str(exc)[:300]})

    _debug("engine_done_final_export_lock", active=final_export_lock.get("active"), accepted=final_export_lock.get("accepted"), reason=final_export_lock.get("reason"), rollback=final_export_lock.get("rollback_to_standard_final"))
    commercial_finish_report = _v62_8_2_apply_measurement_authority_runtime_safe(commercial_finish_report, final_analysis)
    _debug("engine_done_measurement_authority_finish")
    reasons = _safety_risk(before, final_analysis, decision)
    _debug("engine_done_safety_risk", count=len(reasons or []))
    stereo_qc = _stereo_qc_summary(final_analysis)
    _debug("engine_done_stereo_qc_build")
    ab_delta_qc = _ab_delta_qc(before, final_analysis)
    _debug("engine_done_ab_delta_qc_build")
    lra_guard = _lra_guard_summary(before, final_analysis, decision, mode)
    _debug("engine_done_lra_guard_build")
    codec_preview_guard = _codec_preview_guard(before, final_analysis)
    _debug("engine_done_codec_preview_guard_build")
    playback_translation = _playback_translation_report(before, final_analysis)
    _debug("engine_done_playback_translation_build")
    reasons = sorted(set(
        reasons
        + stereo_qc.get("warnings", [])
        + ab_delta_qc.get("warnings", [])
        + lra_guard.get("warnings", [])
        + codec_preview_guard.get("warnings", [])
        + playback_translation.get("warnings", [])
    ))
    adaptive_input_relative_qc_governance = _adaptive_input_relative_qc_governance(
        before, final_analysis, decision, mode, lra_guard, reasons
    )
    try:
        _wr = adaptive_input_relative_qc_governance.get("warning_reclassification", {}) if isinstance(adaptive_input_relative_qc_governance, dict) else {}
        if isinstance(_wr, dict) and _wr.get("adjusted_warnings"):
            reasons = [str(x) for x in _wr.get("adjusted_warnings", [])]
            # v62.9.9.3.1: preserve raw LRA-guard output while making the
            # displayed LRA warnings match adaptive input-relative governance.
            _raw_lra_warnings = [str(x) for x in (lra_guard.get("warnings", []) or [])]
            lra_guard["raw_warnings_before_adaptive_input_relative_qc"] = _raw_lra_warnings
            lra_guard["adaptive_input_relative_reclassification"] = _wr
            _removed = set(str(x) for x in (_wr.get("removed_warnings", []) or []))
            _added_lra = [
                str(x) for x in (_wr.get("added_warnings", []) or [])
                if str(x) in {
                    "source_lra_already_flat",
                    "lra_flat_but_input_relative_loss_small",
                    "source_flat_plus_additional_lra_loss",
                    "mastering_induced_lra_collapse",
                    "lra_low_input_unavailable",
                }
            ]
            _adj_lra = [w for w in _raw_lra_warnings if w not in _removed]
            for _w in _added_lra:
                if _w not in _adj_lra:
                    _adj_lra.append(_w)
            lra_guard["warnings"] = sorted(_adj_lra)
            lra_guard["adaptive_adjusted_lra_warnings"] = sorted(_adj_lra)
    except Exception:
        pass
    lra_warning_semantics = _v6313_3_lra_warning_semantics(lra_guard, adaptive_input_relative_qc_governance, reasons)
    if isinstance(lra_guard, dict):
        lra_guard["warning_semantics_v6313_3"] = lra_warning_semantics
    _debug("engine_done_adaptive_input_relative_qc", active=adaptive_input_relative_qc_governance.get("active") if isinstance(adaptive_input_relative_qc_governance, dict) else None, lra=((adaptive_input_relative_qc_governance.get("lra") or {}) if isinstance(adaptive_input_relative_qc_governance, dict) else {}), warnings=reasons)
    _debug("engine_done_stereo_qc", **stereo_qc)
    _debug("engine_done_lra_guard", **lra_guard)
    _debug("engine_done_codec_preview_guard", **codec_preview_guard)
    _debug("engine_done_playback_translation", overall_score=playback_translation.get("overall_score"), warnings=playback_translation.get("warnings", []))
    _debug("engine_done_ab_delta_qc", warnings=ab_delta_qc.get("warnings", []), deltas=ab_delta_qc.get("deltas", {}))
    _debug("engine_done_final_analysis", reasons=reasons, lufs=final_analysis.get("integrated_lufs"), lra=final_analysis.get("lra_lu"), true_peak=final_analysis.get("approx_true_peak_dbfs"), crest=final_analysis.get("crest_factor_db"), correlation=final_analysis.get("correlation"))
    _v57_2_attach_delivered_final_damage_gate(commercial_finish_report, before, final_analysis, decision, mode, lra_guard, reasons)

    pre_limiter_report = pre_limiter_report or {}
    # Prediction-vs-measurement feedback for calibrating the virtual solver.
    strategy_for_qc = commercial_finish_report.get("final_push_strategy", {}) if isinstance(commercial_finish_report.get("final_push_strategy", {}), dict) else {}
    selected_candidate_for_qc = None
    for _cand in commercial_finish_report.get("candidates", []) or []:
        if isinstance(_cand, dict) and _cand.get("name") == commercial_finish_report.get("selected"):
            selected_candidate_for_qc = _cand
            break
    predicted_lufs = strategy_for_qc.get("expected_lufs_after_push_est")
    predicted_tp = strategy_for_qc.get("recommended_true_peak_ceiling_db")
    measured_lufs = final_analysis.get("integrated_lufs")
    measured_tp = final_analysis.get("approx_true_peak_dbfs")
    measured_crest = final_analysis.get("crest_factor_db")
    actual_vs_predicted_qc = {
        "schema_version": "busy_actual_vs_predicted_qc_v8_5_3_21",
        "prediction_source": "virtual_hot_commercial_solver.final_push_strategy",
        "actual_source": "final_output_analysis_after_strategy_application",
        "selected_candidate": commercial_finish_report.get("selected"),
        "strategy_application_stage": commercial_finish_report.get("strategy_application_stage"),
        "predicted_lufs": predicted_lufs,
        "measured_lufs": measured_lufs,
        "lufs_error_db": round(float(measured_lufs) - float(predicted_lufs), 3) if predicted_lufs is not None and measured_lufs is not None else None,
        "predicted_true_peak_db": predicted_tp,
        "measured_true_peak_db": measured_tp,
        "true_peak_error_db": round(float(measured_tp) - float(predicted_tp), 3) if predicted_tp is not None and measured_tp is not None else None,
        "measured_crest_db": measured_crest,
        "selected_candidate_analysis": (selected_candidate_for_qc or {}).get("analysis", {}) if isinstance(selected_candidate_for_qc, dict) else {},
        "calibration_hint": None,
    }
    try:
        le = actual_vs_predicted_qc.get("lufs_error_db")
        tpe = actual_vs_predicted_qc.get("true_peak_error_db")
        hints = []
        if le is not None and float(le) < -0.6:
            hints.append("virtual_lufs_efficiency_overestimated")
        if le is not None and float(le) > 0.6:
            hints.append("virtual_lufs_efficiency_underestimated")
        if tpe is not None and float(tpe) > 0.25:
            hints.append("true_peak_proxy_underestimated")
        if commercial_finish_report.get("selected") == "base":
            hints.append("no_candidate_selected_or_strategy_application_failed")
        actual_vs_predicted_qc["calibration_hint"] = hints
    except Exception:
        actual_vs_predicted_qc["calibration_hint"] = ["calibration_hint_error"]

    playback_translation_auditor = build_playback_translation_auditor(before, final_analysis, playback_translation)
    post_master_qc_ledger = build_post_master_qc_ledger(before, final_analysis, decision, {
        "remaining_safety_warnings": reasons,
        "rollback_used": pre_limiter_report.get("rollback_used", False),
        "rollback_attempts": pre_limiter_report.get("rollback_attempts", []),
        "loudness_push_governor": pre_limiter_report.get("loudness_push_governor"),
        "commercial_loudness_prep": pre_limiter_report.get("commercial_loudness_prep"),
        "commercial_loudness_finish": commercial_finish_report,
        "virtual_hot_commercial_solver": commercial_finish_report.get("virtual_hot_commercial_solver", {}),
        "blocker_capacity_map": commercial_finish_report.get("blocker_capacity_map", {}),
        "final_push_strategy": commercial_finish_report.get("final_push_strategy", {}),
        "musical_role_identity_map": commercial_finish_report.get("musical_role_identity_map", {}),
        "virtual_premaster_strategy": commercial_finish_report.get("virtual_premaster_strategy", {}),
        "multi_agent_virtual_premaster_strategy_council": commercial_finish_report.get("multi_agent_virtual_premaster_strategy_council", {}),
        "actual_vs_predicted_qc": actual_vs_predicted_qc,
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
        "lra_warning_semantics": lra_warning_semantics if 'lra_warning_semantics' in locals() else {},
        "adaptive_input_relative_qc_governance": adaptive_input_relative_qc_governance,
        "codec_preview_guard": codec_preview_guard,
        "playback_translation_auditor": playback_translation_auditor,
        "lra_recovery_stage": lra_recovery_stage,
        "v6362_dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "v60_deterministic_multistage_limiter": v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {},
        "final_export_lock": final_export_lock,
    })
    final_master_confidence = build_final_master_confidence(
        post_master_qc_ledger,
        playback_translation_auditor,
        codec_preview_guard,
        before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        final_analysis,
    )
    ai_residue_architecture_mapper = build_ai_residue_architecture_report(
        before,
        final_analysis,
        decision,
        pre_limiter_report,
        post_master_qc_ledger,
        playback_translation_auditor,
    )
    _debug("engine_done_post_master_qc_ledger", confidence=post_master_qc_ledger.get("confidence"), decisions=post_master_qc_ledger.get("decisions"), warnings=[w.get("code") for w in post_master_qc_ledger.get("warning_details", []) if isinstance(w, dict)])
    _debug("engine_done_final_master_confidence", confidence=final_master_confidence.get("confidence"), decisions=final_master_confidence.get("decisions"))
    _debug("engine_done_ai_residue_architecture_mapper", confidence=ai_residue_architecture_mapper.get("confidence"), action_level=(ai_residue_architecture_mapper.get("final_action_policy") or {}).get("action_level"), candidates=len(((ai_residue_architecture_mapper.get("shadow_residue_attenuation") or {}).get("candidate_actions") or [])))

    final_master_highlight_review = _run_gemini_final_master_highlight_review_best_effort(
        final,
        sr,
        before if isinstance(before, dict) else {},
        final_analysis if isinstance(final_analysis, dict) else {},
        decision if isinstance(decision, dict) else {},
        commercial_finish_report if isinstance(commercial_finish_report, dict) else {},
        final_export_lock if isinstance(final_export_lock, dict) else {},
        mode,
    )
    final_master_highlight_review_public = {k: v for k, v in final_master_highlight_review.items() if k not in {"raw_text"}}
    if isinstance(commercial_finish_report, dict):
        commercial_finish_report["final_master_highlight_review"] = final_master_highlight_review_public
    _debug(
        "engine_done_gemini_final_master_highlight_review",
        enabled=final_master_highlight_review_public.get("enabled"),
        available=final_master_highlight_review_public.get("available"),
        verdict=final_master_highlight_review_public.get("verdict"),
        commercial_readiness=final_master_highlight_review_public.get("commercial_readiness"),
        reason=final_master_highlight_review_public.get("reason"),
    )

    report = {
        "engine_version": V6362_MASTERING_ENGINE_VERSION,
        "mastering_report_schema_version": "busy_master_report_v8_5_3_64_0_4_commercial_maximum_density_push_foundation",
        "runtime_intensity_lock": runtime_intensity_lock,
        "two_stage_limiting": True,
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": pre_limiter_report.get("original_input_analysis_before_pre_clean", {}),
        "residue_pre_clean_stage": pre_limiter_report.get("residue_pre_clean_stage", {}),
        "v6316_standard_final_loudness_iteration": standard_final_loudness_iteration if 'standard_final_loudness_iteration' in locals() else {},
        "pre_limiter_report": pre_limiter_report,
        "v6316_mastering_chain_reconnect": {
            "schema_version": "busy_v6316_mastering_chain_reconnect",
            "active": True,
            "pre_limiter_handoff": pre_limiter_report.get("v6316_mastering_chain_reconnect", {}) if isinstance(pre_limiter_report, dict) else {},
            "commercial_finish_evaluated": bool(isinstance(commercial_finish_report, dict) and commercial_finish_report.get("retry_attempted") is not None),
            "commercial_finish_selected": commercial_finish_report.get("selected") if isinstance(commercial_finish_report, dict) else None,
            "v60_active": (v60_deterministic_multistage_limiter or {}).get("active") if isinstance(v60_deterministic_multistage_limiter, dict) else None,
            "v60_applied": (v60_deterministic_multistage_limiter or {}).get("applied") if isinstance(v60_deterministic_multistage_limiter, dict) else None,
            "v61_attempted": (pre_limiter_report.get("v61_render_measured_feedback_optimizer", {}) or {}).get("attempted") if isinstance(pre_limiter_report, dict) and isinstance(pre_limiter_report.get("v61_render_measured_feedback_optimizer"), dict) else None,
            "policy": "v63.1.6.2.1 aligns the reconnected mastering chain with the deep-research design: true pre-limiter handoff, commercial floor recovery after safety limiting, v60 on current final reference, v61 post-safety candidate QC, and delivered-buffer authority.",
        },
        "v62_8_mastering_blocker_preparation": decision.get("v62_8_mastering_blocker_preparation", pre_limiter_report.get("v62_8_mastering_blocker_preparation", {})) if isinstance(decision, dict) else pre_limiter_report.get("v62_8_mastering_blocker_preparation", {}),
        "lgcca_influence_routing": decision.get("lgcca_influence_routing", {}) if isinstance(decision, dict) else {},
        "lgcca_deep_wiring": decision.get("lgcca_deep_wiring", {}) if isinstance(decision, dict) else {},
        "action_governance": decision.get("action_governance", {}) if isinstance(decision, dict) else {},
        "existing_process_research_governance": decision.get("existing_process_research_governance", {}) if isinstance(decision, dict) else {},
        "v62_9_9_research_governance": decision.get("v62_9_9_research_governance", {}) if isinstance(decision, dict) else {},
        "microdynamics_lra_shape_recovery": (decision.get("v62_9_9_research_governance", {}) or {}).get("microdynamics_lra_shape_recovery", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "section_aware_spectral_consistency": (decision.get("v62_9_9_research_governance", {}) or {}).get("section_aware_spectral_consistency", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "musical_texture_preservation_gate": (decision.get("v62_9_9_research_governance", {}) or {}).get("musical_texture_preservation_gate", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "translation_aware_low_end_governor": (decision.get("v62_9_9_research_governance", {}) or {}).get("translation_aware_low_end_governor", {}) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {},
        "ai_call_governance": decision.get("ai_call_governance", {}) if isinstance(decision, dict) else {},
        "output_analysis": final_analysis,
        "working_stage_gain_db": pre_limiter_report.get("working_stage_gain_db"),
        "final_true_peak_gain_db": round(final_true_peak_gain_db, 3),
        "true_peak_ceiling_attenuation": true_peak_ceiling_attenuation if isinstance(locals().get("true_peak_ceiling_attenuation"), dict) else {},
        "final_safety_ceiling_enforcement": final_safety_ceiling_enforcement,
        "final_export_lock": final_export_lock,
        "final_true_peak_target_dbtp": FINAL_TRUE_PEAK_CEILING_DB,
        "requested_oversample_factor": adaptive_quality_report.get("requested_oversample"),
        "oversample_factor": os_factor,
        "adaptive_quality": adaptive_quality_report,
        "loudness_push_governor": pre_limiter_report.get("loudness_push_governor"),
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
        "lra_warning_semantics": lra_warning_semantics if 'lra_warning_semantics' in locals() else {},
        "adaptive_input_relative_qc_governance": adaptive_input_relative_qc_governance,
        "lra_recovery_stage": lra_recovery_stage,
        "v6361_drum_transient_punch": v6361_drum_transient_punch if 'v6361_drum_transient_punch' in locals() else {},
        "drum_transient_punch": v6361_drum_transient_punch if 'v6361_drum_transient_punch' in locals() else {},
        "v6404_commercial_maximum_density": ({k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"} if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else {}),
        "commercial_maximum_density": ({k: v for k, v in v6404_commercial_maximum_density.items() if k != "selected_analysis"} if 'v6404_commercial_maximum_density' in locals() and isinstance(v6404_commercial_maximum_density, dict) else {}),
        "mastering_ownership": _v6351_select_report_ownership(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "commercial_push_safety_gate": _v6351_select_report_gate(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "selected_loudness_class": (_v6351_select_report_ownership(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}) or {}).get("selected_loudness_class"),
        "push_authority": (_v6351_select_report_ownership(decision if isinstance(decision, dict) else {}, commercial_finish_report if isinstance(commercial_finish_report, dict) else {}, v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {}) or {}).get("push_authority"),
        "v6362_dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "dml_ownership_contract": ((v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("v6362_dml_ownership_contract"), dict) else ((v60_deterministic_multistage_limiter.get("acceptance_gate", {}) or {}).get("v6362_dml_ownership_contract") if isinstance(v60_deterministic_multistage_limiter, dict) and isinstance(v60_deterministic_multistage_limiter.get("acceptance_gate"), dict) else {})) if 'v60_deterministic_multistage_limiter' in locals() else {}),
        "v60_deterministic_multistage_limiter": v60_deterministic_multistage_limiter if 'v60_deterministic_multistage_limiter' in locals() else {},
        "v61_render_measured_feedback_optimizer": pre_limiter_report.get("v61_render_measured_feedback_optimizer", {}) if isinstance(pre_limiter_report, dict) else {},
        "multi_pass_optimizer": optimizer_report,
        "adaptive_residue_light_control": experimental_residue_light_control,
        "experimental_residue_light_control": experimental_residue_light_control,
        "adaptive_hot_commercial": {
            "enabled": bool(_experimental_hot_commercial_enabled(mode)),
            "auto_residue_control_enabled": bool(_env_bool("BUSY_RESIDUE_PRE_CLEAN", True)),
            "target_lufs_env_override": os.environ.get("BUSY_EXPERIMENTAL_HOT_TARGET_LUFS"),
            "strength": _env_float("BUSY_ADAPTIVE_RESIDUE_STRENGTH", _env_float("BUSY_AI_RESIDUE_DIRECT_STRENGTH", 1.0, 0.20, 2.0), 0.20, 2.0),
            "policy": "hot commercial candidate testing after one-time pre-master cleanup; no repeated post-master residue trimming",
        },
        "experimental_hot_commercial": {
            "enabled": bool(_experimental_hot_commercial_enabled(mode)),
            "compat_alias": True,
            "policy": "compat alias; see adaptive_hot_commercial",
        },
        "commercial_loudness_finish": commercial_finish_report,
        "final_perceptual_ab_arbiter": (commercial_finish_report.get("v62_9_9_final_perceptual_ab_arbiter") if isinstance(commercial_finish_report, dict) else None) or (((decision.get("v62_9_9_research_governance", {}) or {}).get("final_perceptual_ab_arbiter_policy", {})) if isinstance(decision, dict) and isinstance(decision.get("v62_9_9_research_governance"), dict) else {}),
        "v58_spectral_conditioning_plan": decision.get("v58_spectral_conditioning_plan", {}) if isinstance(decision, dict) else {},
        "v58_spectral_conditioning_render_report": (pre_limiter_report.get("processing_report", {}) or {}).get("v58_spectral_conditioning", {}) if isinstance(pre_limiter_report.get("processing_report", {}), dict) else {},
        "v59_lf_crest_source_conditioning_plan": decision.get("v59_lf_crest_source_conditioning_plan", {}) if isinstance(decision, dict) else {},
        "v59_lf_crest_source_conditioning_render_report": (pre_limiter_report.get("processing_report", {}) or {}).get("v59_lf_crest_source_conditioning", {}) if isinstance(pre_limiter_report.get("processing_report", {}), dict) else {},
        "final_master_highlight_review": final_master_highlight_review_public,
        "gemini_final_master_highlight_review": final_master_highlight_review_public,
        "virtual_hot_commercial_solver": commercial_finish_report.get("virtual_hot_commercial_solver", {}),
        "representative_section_map": commercial_finish_report.get("representative_section_map", {}),
        "virtual_limiter_sweep": (commercial_finish_report.get("virtual_hot_commercial_solver", {}) or {}).get("virtual_limiter_sweep", {}) if isinstance(commercial_finish_report.get("virtual_hot_commercial_solver", {}), dict) else {},
        "blocker_capacity_map": commercial_finish_report.get("blocker_capacity_map", {}),
        "blocker_solution_plan": commercial_finish_report.get("blocker_solution_plan", {}),
        "final_push_strategy": commercial_finish_report.get("final_push_strategy", {}),
        "musical_role_identity_map": commercial_finish_report.get("musical_role_identity_map", {}),
        "virtual_premaster_strategy": commercial_finish_report.get("virtual_premaster_strategy", {}),
        "multi_agent_virtual_premaster_strategy_council": commercial_finish_report.get("multi_agent_virtual_premaster_strategy_council", {}),
        "actual_vs_predicted_qc": actual_vs_predicted_qc,
        "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        "commercial_loudness_result": {
            "target_lufs": commercial_finish_report.get("target_lufs"),
            "commercial_floor_lufs": commercial_finish_report.get("commercial_floor_lufs"),
            "final_lufs": final_analysis.get("integrated_lufs"),
            "final_lufs_authority": commercial_finish_report.get("final_lufs_authority"),
            "final_lufs_stereo": commercial_finish_report.get("final_lufs_stereo"),
            "final_lufs_mono_downmix": commercial_finish_report.get("final_lufs_mono_downmix"),
            "target_met": commercial_finish_report.get("target_met"),
            "target_reached": commercial_finish_report.get("target_reached"),
            "retry_attempted": commercial_finish_report.get("retry_attempted"),
            "reason": commercial_finish_report.get("reason"),
            "retry_blocked_by": commercial_finish_report.get("retry_blocked_by"),
            "policy": "commercial loudness floor is checked, but unresolved new problem sets override raw loudness; warning-only hot pass is disabled",
        },
        "codec_preview_guard": codec_preview_guard,
        "playback_translation": playback_translation,
        "playback_translation_auditor": playback_translation_auditor,
        "post_master_qc_explainer": post_master_qc_ledger,
        "rollback_ledger": post_master_qc_ledger,
        "final_master_confidence": final_master_confidence,
        "ai_residue_architecture_mapper": ai_residue_architecture_mapper,
        "ai_sound_architecture_analysis": ai_residue_architecture_mapper.get("ai_sound_architecture_analysis", {}),
        "residue_component_separation": ai_residue_architecture_mapper.get("residue_component_separation", {}),
        "musicality_protection_map": ai_residue_architecture_mapper.get("musicality_protection_map", {}),
        "shadow_residue_attenuation": ai_residue_architecture_mapper.get("shadow_residue_attenuation", {}),
        "direct_residue_action_gate": ai_residue_architecture_mapper.get("direct_residue_action_gate", {}),
        "ai_residue_final_user_summary": ai_residue_architecture_mapper.get("final_user_summary", {}),
        "dither_noise_shaping": dither_report,
        "rollback_used": pre_limiter_report.get("rollback_used", False),
        "rollback_attempts": pre_limiter_report.get("rollback_attempts", []),
        "remaining_safety_warnings": reasons,
        "risk_metrics": _risk_metrics(before, final_analysis),
        "character_scale_initial": pre_limiter_report.get("character_scale_initial"),
        "genre_chain": decision.get("genre_chain", {}),
        "processing_report": pre_limiter_report.get("processing_report", {}),
        "decision_summary": pre_limiter_report.get("decision_summary", {
            "detected_style": decision.get("detected_style") or decision.get("genre_id"),
            "selected_profile": decision.get("selected_profile"),
            "style_confidence": decision.get("style_confidence"),
            "boldness_level": decision.get("boldness_level", "auto"),
            "profile_blend": decision.get("profile_blend"),
            "mastering_intent": decision.get("mastering_intent"),
            "reference_v7": decision.get("reference_v7"),
        }),
    }
    report = _v62_9_4_1_attach_top_level_final_metric_aliases(report, final_analysis, commercial_finish_report)
    report = _v62_9_9_3_2_reconcile_adaptive_input_relative_qc_report(report, decision, mode)
    report = _v6314_4_finalize_floor_tolerance_and_handoff_markers(report, decision)
    report = _v6362_8_finalize_report_authority(report, decision)
    _debug("engine_done", output_lufs=final_analysis.get("integrated_lufs"), output_true_peak=final_analysis.get("approx_true_peak_dbfs"), oversample_factor=os_factor, adaptive_quality=adaptive_quality_report)
    return np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0), report
