from __future__ import annotations

import copy
import gc
import json
import os
import resource
from datetime import datetime, timezone
from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt, lfilter, resample_poly
try:
    import pyloudnorm as pyln
except Exception:  # optional dependency during local smoke tests
    pyln = None

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

WORKING_STAGE_LUFS = -18.0
FINAL_TRUE_PEAK_CEILING_DB = float(os.environ.get("BUSY_TRUE_PEAK_CEILING_DB", "-0.1"))


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


def _measure_lufs(y: np.ndarray, sr: int) -> float | None:
    mono = np.mean(y, axis=1)
    if pyln is not None:
        try:
            meter = pyln.Meter(sr)
            return float(meter.integrated_loudness(mono))
        except Exception:
            pass
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    return 20 * np.log10(max(rms, 1e-12)) - 3.0


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

    # Low-side bloom is the safest direct cleanup and often unlocks limiter headroom.
    low_side_amount = 0.0
    if side_low >= _env_float("BUSY_AI_RESIDUE_LOW_SIDE_TRIGGER", 0.08, 0.03, 0.25):
        low_side_amount = float(np.clip(0.52 + side_low * 2.15 * strength, 0.50, 0.92))
        static_moves.append({"type": "bell", "freq": 125, "gain_db": -0.45 * strength, "q": 0.85, "target": "side", "source": "experimental_low_side_bloom_control"})
        dynamic_moves.append({"type": "bell", "freq": 92, "gain_db": -0.85 * strength, "q": 0.85, "target": "side", "source": "experimental_low_side_bloom_control"})
        actions.append({"action": "low_side_bloom_control", "amount": round(low_side_amount, 3), "trigger_side_low": round(side_low, 5)})

    # 7-12 kHz side shimmer / brittle top.  This is where Suno fizz often becomes
    # exposed by limiting, but keep it side-weighted unless the full high band is risky.
    br_conf = conf(br)
    if br_conf >= min_conf or side_high >= 0.42 or side_high_risk >= 0.32:
        cut = float(np.clip(0.45 + max(0.0, br_conf - min_conf) * 2.4 + max(0.0, side_high - 0.42) * 1.1, 0.35, 1.55)) * strength
        dynamic_moves.append({"type": "bell", "freq": 9200, "gain_db": -cut, "q": 1.45, "target": "side", "source": "experimental_side_high_fizz_control"})
        actions.append({"action": "side_high_fizz_control", "band": "brilliance_7000_12000", "attenuation_db": round(cut, 3), "confidence": round(br_conf, 4)})

    # 12-18 kHz synthetic air.  If side ratio is low, use a very gentle stereo
    # guard because the residue may be centered/high-wide rather than side-only.
    air_conf = conf(air)
    if air_conf >= min_conf:
        side_pref = side_ratio(air) >= 0.08 or side_high_risk >= 0.35
        target = "side" if side_pref else "stereo"
        cut = float(np.clip(0.36 + max(0.0, air_conf - min_conf) * 2.0, 0.30, 1.20)) * strength
        dynamic_moves.append({"type": "bell", "freq": 14500, "gain_db": -cut, "q": 0.95, "target": target, "source": "experimental_synthetic_air_control"})
        static_moves.append({"type": "high_shelf", "freq": 13200, "gain_db": -0.16 * strength, "q": 0.70, "target": target, "source": "experimental_synthetic_air_control"})
        actions.append({"action": "synthetic_air_control", "band": "air_12000_18000", "target": target, "attenuation_db": round(cut, 3), "confidence": round(air_conf, 4)})

    # 18k+ ultra-air hash.  Keep this subtle; it is mostly about avoiding a brittle
    # ultrasonic/near-Nyquist edge before pushing the limiter harder.
    ultra_conf = conf(ultra)
    if ultra_conf >= min_conf:
        target = "side" if side_ratio(ultra) >= 0.06 or side_high_risk >= 0.35 else "stereo"
        cut = float(np.clip(0.28 + max(0.0, ultra_conf - min_conf) * 1.6, 0.25, 0.95)) * strength
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
    if pres_conf >= min_conf + 0.12 and harsh > -3.2 and vocal_role < 0.72:
        cut = float(np.clip(0.20 + max(0.0, pres_conf - min_conf) * 1.2, 0.18, 0.75)) * strength
        dynamic_moves.append({"type": "bell", "freq": 5200, "gain_db": -cut, "q": 1.45, "target": "stereo", "source": "experimental_presence_edge_control"})
        actions.append({"action": "presence_edge_control", "band": "presence_4000_7000", "attenuation_db": round(cut, 3), "confidence": round(pres_conf, 4)})

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
            },
        }

    return {
        "active": True,
        "schema_version": "busy_adaptive_direct_residue_light_control_v8_5_3_14",
        "stage": stage,
        "mode": "adaptive_commercial_residue_light_control",
        "strength": round(strength, 3),
        "min_confidence": round(min_conf, 3),
        "low_side_mono_amount": round(low_side_amount, 3),
        "static_moves": static_moves,
        "dynamic_moves": dynamic_moves,
        "actions": actions,
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
        "schema_version": "busy_residue_pre_clean_stage_v8_5_3_16",
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
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render a hot-commercial candidate without letting the limiter erase the gain.

    If a high-gain candidate goes straight into a transparent limiter, peak
    overshoot can make the limiter turn the whole song down.  This path records
    every loudness stage and adds peak absorption/density recovery so the virtual
    strategy is actually translated into final LUFS.
    """
    trace: list[dict[str, Any]] = []
    os_factor = _qc_oversample_factor()

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
        drive = float(np.clip(0.65 + over_db * 0.22, 0.75, 3.25))
        mix = float(np.clip(0.22 + over_db * 0.055, 0.22, 0.78))
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
        lookahead_ms=0.8,
        release_ms=115.0,
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
    for idx in range(4):
        if best_lufs >= min(float(target_lufs) - 0.15, float(floor_lufs) + 0.75):
            break
        gap = max(0.0, min(float(target_lufs), float(floor_lufs) + 1.35) - best_lufs)
        if gap < 0.18:
            break
        add_gain = float(np.clip(gap * 0.72, 0.25, 1.65))
        trial = apply_gain_db(best, add_gain)
        clip_drive = float(np.clip(0.85 + add_gain * 0.65 + idx * 0.20, 0.85, 3.60))
        clip_mix = float(np.clip(0.30 + add_gain * 0.10 + idx * 0.055, 0.30, 0.82))
        trial = soft_clip(trial, drive_db=clip_drive, mix=clip_mix)
        trial = oversampled_limit_chunked(
            trial,
            sr,
            ceiling_db=ceiling_db,
            oversample=os_factor,
            lookahead_ms=0.65,
            release_ms=90.0,
        )
        trial, trial_tp_gain = _normalize_to_true_peak(trial, ceiling_db, sr=sr, oversample=os_factor)
        trial_an = analyze_audio_fast_qc(trial, sr, true_peak_oversample=os_factor)
        trial_lufs = float(trial_an.get("integrated_lufs", -99.0) or -99.0)
        trial_crest = float(trial_an.get("crest_factor_db", 99.0) or 99.0)
        trial_warnings = _safety_risk(before, trial_an, decision)
        trial_hard = _commercial_loudness_candidate_hard_blocks(before, trial_an, decision, profile, trial_warnings)
        improved = trial_lufs > prev_lufs + 0.12
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
            "accepted": bool(improved and not trial_hard),
            "stop_if_rejected_reason": "no_lufs_improvement_or_hard_block",
        }
        recovery_events.append(event)
        trace.append({"stage": f"density_recovery_{idx+1}", **_analysis_summary_for_loudness(trial_an)})
        if improved and not trial_hard:
            best = trial
            best_an = trial_an
            best_lufs = trial_lufs
            prev_lufs = trial_lufs
        else:
            break

    limiter_loss_db = float(start_lufs - best_lufs) if np.isfinite(start_lufs) and np.isfinite(best_lufs) else 0.0
    report = {
        "schema_version": "busy_hot_candidate_render_trace_v8_5_3_20",
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
    plan = _commercial_loudness_prep_plan(before, decision, mode)
    d = copy.deepcopy(decision or {})
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
) -> tuple[np.ndarray, dict[str, Any]]:
    d = _normalize_decision(decision)
    defaults = _mode_defaults(mode)
    chain = _genre_chain_params(d, mode)
    report: dict[str, Any] = {
        "active": False,
        "actions": [],
        "stages": [],
        "genre_chain": chain,
        "spatial_fx_mastering": d.get("spatial_fx_mastering_plan", {}),
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

    dyn_intensity = float(d.get("dynamic_eq_intensity", 1.0)) * min(1.45, max(0.70, character_scale)) * float(chain.get("dynamic_eq_multiplier", 1.0))
    dyn_intensity = float(np.clip(dyn_intensity, 0.45, 1.95))
    out = dynamic_eq(out, sr, _collect_moves(d, "dynamic_eq"), intensity=dyn_intensity)
    report["stages"].append(f"dynamic_eq_intensity_{dyn_intensity:.2f}")

    out, vocal_report = _vocal_lead_protection(out, sr, d)
    report["vocal_lead_protection"] = vocal_report
    if vocal_report.get("active"):
        report["stages"].append("vocal_lead_protection")

    mb = d.get("multiband", {}) or {}
    mb_cfg = d.get("multiband_compression", {}) or mb.get("multiband_compression", {}) or {}
    band_settings = mb_cfg.get("bands", []) if isinstance(mb_cfg, dict) else []
    mb_amount = float(mb.get("amount", defaults["mb_amount"])) * mb_amount_scale * float(chain.get("multiband_multiplier", 1.0))
    mb_amount = float(np.clip(mb_amount, 0.25, 1.85))
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
    out, iter_report = _loudness_iteration(out, sr, target_lufs=target_lufs, ceiling_db=ceiling_db, limiter=d.get("limiter", {}), max_iters=7)
    iter_report["commercial_loudness_prep_extra_push_db"] = round(loudprep_extra_push, 3)
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
    _debug("engine_start", sr=sr, shape=getattr(y, "shape", None), dtype=str(getattr(y, "dtype", "")), mode=mode)
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

    staged, stage_gain_db = _stage_to_working_lufs(y, sr, WORKING_STAGE_LUFS)
    character_scale = _boldness_scale(decision, mode)
    _debug("engine_done_working_stage", stage_gain_db=round(stage_gain_db, 3), character_scale=round(character_scale, 3), staged_shape=getattr(staged, "shape", None))

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

    # Multi-step rollback: reduce character/loudness first, then restore a hair of transient if over-crushed.
    if reasons:
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
    _debug("engine_start_true_peak_normalize", oversample=os_factor, target_db=FINAL_TRUE_PEAK_CEILING_DB)
    final, final_true_peak_gain_db = _normalize_to_true_peak(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=os_factor)
    _debug("engine_done_true_peak_normalize", gain_db=round(final_true_peak_gain_db, 3))

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
    commercial_source_for_finish = pre_limiter if "pre_limiter" in locals() else final
    commercial_candidate, commercial_finish_report = _commercial_loudness_finish_pass(commercial_source_for_finish, sr, before, decision, mode)
    commercial_selected = bool(commercial_finish_report.get("selected") and commercial_finish_report.get("selected") != "base")
    commercial_finish_report["strategy_application_stage"] = "pre_limiter_before_final_safety_limiter"
    # v8.5.3.20: never replace the already rendered standard final with a
    # commercial candidate that is quieter.  The finish pass is evaluated against
    # the pre-limiter source, so its internal base can be much quieter than the
    # standard final.  This guard prevents a "selected" pre-limiter candidate from
    # making the delivered master smaller.
    standard_final_reference = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    commercial_finish_report["standard_final_reference_analysis"] = _analysis_summary_for_loudness(standard_final_reference)
    try:
        std_lufs = float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)
        cand_lufs = float((commercial_finish_report.get("output") or {}).get("integrated_lufs", -99.0) or -99.0)
    except Exception:
        std_lufs, cand_lufs = -99.0, -99.0
    if commercial_selected and cand_lufs < std_lufs + 0.12 and not bool(commercial_finish_report.get("target_met")):
        commercial_finish_report["candidate_rejected_against_standard_final"] = True
        commercial_finish_report["candidate_rejection_reason"] = "candidate_not_louder_than_standard_final"
        commercial_finish_report["candidate_vs_standard_lufs_delta_db"] = round(float(cand_lufs - std_lufs), 3)
        commercial_selected = False
    commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = bool(commercial_selected)
    if commercial_selected:
        final = commercial_candidate
    else:
        commercial_finish_report["fallback_final_source"] = "standard_final_limiter_output"
    _debug(
        "engine_done_commercial_loudness_finish",
        active=commercial_finish_report.get("active"),
        target_met=commercial_finish_report.get("target_met"),
        selected=commercial_finish_report.get("selected"),
        reason=commercial_finish_report.get("reason"),
        output=(commercial_finish_report.get("output") or {}).get("integrated_lufs"),
        strategy_application_stage=commercial_finish_report.get("strategy_application_stage"),
    )

    final, dither_report = _apply_output_dither(final, bit_depth=24)
    if dither_report.get("active"):
        _debug("engine_done_dither_noise_shaping", **dither_report)

    _debug("engine_start_final_analysis")
    final_analysis = analyze_audio(final, sr)
    reasons = _safety_risk(before, final_analysis, decision)
    stereo_qc = _stereo_qc_summary(final_analysis)
    ab_delta_qc = _ab_delta_qc(before, final_analysis)
    lra_guard = _lra_guard_summary(before, final_analysis, decision, mode)
    codec_preview_guard = _codec_preview_guard(before, final_analysis)
    playback_translation = _playback_translation_report(before, final_analysis)
    reasons = sorted(set(
        reasons
        + stereo_qc.get("warnings", [])
        + ab_delta_qc.get("warnings", [])
        + lra_guard.get("warnings", [])
        + codec_preview_guard.get("warnings", [])
        + playback_translation.get("warnings", [])
    ))
    _debug("engine_done_stereo_qc", **stereo_qc)
    _debug("engine_done_lra_guard", **lra_guard)
    _debug("engine_done_codec_preview_guard", **codec_preview_guard)
    _debug("engine_done_playback_translation", overall_score=playback_translation.get("overall_score"), warnings=playback_translation.get("warnings", []))
    _debug("engine_done_ab_delta_qc", warnings=ab_delta_qc.get("warnings", []), deltas=ab_delta_qc.get("deltas", {}))
    _debug("engine_done_final_analysis", reasons=reasons, lufs=final_analysis.get("integrated_lufs"), lra=final_analysis.get("lra_lu"), true_peak=final_analysis.get("approx_true_peak_dbfs"), crest=final_analysis.get("crest_factor_db"), correlation=final_analysis.get("correlation"))

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
        "schema_version": "busy_actual_vs_predicted_qc_v8_5_3_20",
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
        "actual_vs_predicted_qc": actual_vs_predicted_qc,
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
        "codec_preview_guard": codec_preview_guard,
        "playback_translation_auditor": playback_translation_auditor,
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

    report = {
        "engine_version": "busy_auto_mastering_v8_5_3_20_candidate_limiter_trace_fix",
        "mastering_report_schema_version": "busy_master_report_v8_5_3_20_candidate_limiter_trace_fix",
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": original_before,
        "residue_pre_clean_stage": residue_pre_clean_stage,
        "output_analysis": final_analysis,
        "working_stage_gain_db": round(stage_gain_db, 3),
        "final_true_peak_gain_db": round(final_true_peak_gain_db, 3),
        "final_true_peak_target_dbtp": FINAL_TRUE_PEAK_CEILING_DB,
        "requested_oversample_factor": adaptive_quality_report.get("requested_oversample"),
        "oversample_factor": os_factor,
        "adaptive_quality": adaptive_quality_report,
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
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
        "multi_pass_optimizer": optimizer_report,
        "commercial_loudness_finish": commercial_finish_report,
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
            "target_met": commercial_finish_report.get("target_met"),
            "target_reached": commercial_finish_report.get("target_reached"),
            "retry_attempted": commercial_finish_report.get("retry_attempted"),
            "reason": commercial_finish_report.get("reason"),
            "retry_blocked_by": commercial_finish_report.get("retry_blocked_by"),
            "policy": "commercial loudness floor is checked before accepting a quiet master",
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
    _debug("engine_start_pre_limiter_prepare", sr=sr, shape=getattr(y, "shape", None), dtype=str(getattr(y, "dtype", "")), mode=mode)
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

    staged, stage_gain_db = _stage_to_working_lufs(y, sr, WORKING_STAGE_LUFS)
    character_scale = _boldness_scale(render_decision, mode)
    _debug("engine_done_working_stage", stage_gain_db=round(stage_gain_db, 3), character_scale=round(character_scale, 3), staged_shape=getattr(staged, "shape", None))

    prepare_report = {
        "engine_version": "busy_auto_mastering_v8_5_3_16_stage_a2a_pre_clean_prepare",
        "stage": "pre_limiter_prepare",
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": original_before,
        "residue_pre_clean_stage": residue_pre_clean_stage,
        "working_stage_gain_db": round(stage_gain_db, 3),
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
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
    _debug("engine_start_pre_limiter_render", sr=sr, shape=getattr(staged, "shape", None), dtype=str(getattr(staged, "dtype", "")), mode=mode)
    before = prepare_report.get("input_analysis", {}) or {}
    governor_report = prepare_report.get("loudness_push_governor", {}) or {}
    commercial_loudness_prep = prepare_report.get("commercial_loudness_prep", {}) or {}
    stage_gain_db = float(prepare_report.get("working_stage_gain_db", 0.0) or 0.0)
    character_scale = float(prepare_report.get("character_scale_initial", _boldness_scale(decision, mode)) or 1.0)

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

    if reasons:
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
            else:
                cand = None
            gc.collect()
        _, final, final_analysis, final_proc, reasons = best
        _debug("engine_done_rollback", remaining_reasons=reasons, selected_lufs=final_analysis.get("integrated_lufs"), selected_true_peak=final_analysis.get("approx_true_peak_dbfs"), selected_crest=final_analysis.get("crest_factor_db"))

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

    report = {
        "engine_version": "busy_auto_mastering_v8_5_3_16_stage_a2b_pre_limiter",
        "stage": "pre_limiter",
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": prepare_report.get("original_input_analysis_before_pre_clean", {}),
        "residue_pre_clean_stage": prepare_report.get("residue_pre_clean_stage", {}),
        "pre_limiter_analysis_fast": final_analysis,
        "working_stage_gain_db": round(stage_gain_db, 3),
        "loudness_push_governor": governor_report,
        "commercial_loudness_prep": commercial_loudness_prep,
        "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        "spatial_fx_mastering": prepare_report.get("spatial_fx_mastering", decision.get("spatial_fx_mastering_plan", {})),
        "rollback_used": rollback_used,
        "rollback_attempts": rollback_attempts,
        "remaining_safety_warnings_pre_limiter": reasons,
        "risk_metrics_pre_limiter": _risk_metrics(before, final_analysis),
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
    _debug("engine_done_pre_limiter_chain", lufs=final_analysis.get("integrated_lufs"), true_peak=final_analysis.get("approx_true_peak_dbfs"), crest=final_analysis.get("crest_factor_db"))
    return np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0), report


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
    _debug("engine_start_pre_limiter_chain", sr=sr, shape=getattr(y, "shape", None), dtype=str(getattr(y, "dtype", "")), mode=mode)
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
    out, _ = _normalize_to_true_peak(out, ceiling_db, sr=sr, oversample=_qc_oversample_factor())
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
            rep = {
                "name": name,
                "score": score,
                "warnings": warnings,
                "lufs": cand_analysis.get("integrated_lufs"),
                "true_peak": cand_analysis.get("approx_true_peak_dbfs"),
                "crest": cand_analysis.get("crest_factor_db"),
            }
            reports.append(rep)
            _debug("optimizer_done_candidate", **rep)
            if score > best_score + 0.75:
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
    max_push_db = _env_float("BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB", max_push_db, 0.0, 6.0 if hot_exp else 3.2)
    crest_min = _env_float("BUSY_COMMERCIAL_FINISH_CREST_MIN_DB", crest_min, 4.2, 9.0)
    hard_crest_min = min(crest_min - 0.25, _env_float("BUSY_COMMERCIAL_FINISH_HARD_CREST_MIN_DB", hard_crest_min, 4.0, 8.5))

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

    # Keep the floor below the target; target/floor are negative LUFS values.
    if floor_lufs > target_lufs - 0.25:
        floor_lufs = target_lufs - 0.45
    # Avoid accepting obviously quiet results when the private DB asks for a hot
    # master, but do not make a missing DB fallback dangerously aggressive.
    if source.startswith("reference_v7"):
        floor_lufs = max(floor_lufs, target_lufs - _env_float("BUSY_COMMERCIAL_FINISH_MAX_TARGET_FLOOR_GAP_DB", 1.45, 0.45, 3.0))

    return {
        "active": True,
        "schema_version": "busy_commercial_loudness_finish_v8_5_3_20_candidate_limiter_trace_fix" if hot_exp else "busy_commercial_loudness_finish_v8_5_3_9_profile_stack_db_driven",
        "experimental_hot_commercial": bool(hot_exp),
        "adaptive_hot_commercial": bool(hot_exp),
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
    return {
        "integrated_lufs": analysis.get("integrated_lufs"),
        "true_peak_dbtp": analysis.get("approx_true_peak_dbfs"),
        "crest_db": analysis.get("crest_factor_db"),
        "correlation": analysis.get("correlation"),
        "side_low_ratio": ms.get("side_ratio_lowband"),
        "side_high_ratio": ms.get("side_ratio_8k_14k"),
        "harshness_index_db": q.get("harshness_index_db"),
        "air_index_db": q.get("air_index_db"),
        "lra_lu": _analysis_lra_lu(analysis),
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
        # LRA warnings are not always fatal for club masters, but if crest is also
        # near the floor, the candidate is probably flattened rather than finished.
        if crest < float(profile.get("crest_min_db", 5.6) or 5.6) + 0.35:
            hard.append("commercial_lra_flat_with_low_crest")
    return sorted(set(hard))


def _condition_for_commercial_push(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any],
    current_analysis: dict[str, Any],
    push_db: float,
    profile: dict[str, Any],
    virtual_hot_plan: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Control likely limiter failure points before adding more level."""
    out = y
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
        out = multiband_compress(out, sr, mode="hot", amount=float(np.clip(0.28 + max(0.0, push_db) * 0.08, 0.28, 0.62)))
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
    out = soft_clip(out, drive_db=clip_drive, mix=clip_mix)
    notes.append("experimental_hot_clip_before_commercial_push" if experimental else "micro_clip_before_commercial_push")
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), {
        "notes": notes,
        "static_move_count": len(moves),
        "dynamic_move_count": len(dyn),
        "micro_clip_drive_db": round(clip_drive, 3),
        "micro_clip_mix": round(clip_mix, 4),
    }


def _commercial_loudness_finish_pass(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any],
    decision: dict[str, Any],
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Final repair-then-push pass for Auto Commercial Master.

    If the result is below its family commercial floor, try a small set of
    progressively louder candidates.  Loudness is only accepted after QC; safety
    remains the hard stop, but quiet masters are now reported as target misses.
    """
    profile = _commercial_loudness_finish_profile(decision, mode)
    if not profile.get("active"):
        return y, {**profile, "target_met": None, "retry_attempted": False}

    base = analyze_audio_fast_qc(y, sr, true_peak_oversample=_qc_oversample_factor())
    base_lufs = float(base.get("integrated_lufs", -99.0) or -99.0)
    target = float(profile.get("target_lufs", -8.9) or -8.9)
    floor = float(profile.get("commercial_floor_lufs", -9.7) or -9.7)
    target_met = bool(base_lufs >= floor)
    report: dict[str, Any] = {
        **profile,
        "input": _analysis_summary_for_loudness(base),
        "target_met_before_retry": target_met,
        "target_met": target_met,
        "retry_attempted": False,
        "selected": "base",
        "candidates": [],
    }

    # v8.5.3.16: before deciding how hard to actually push the full song, run a
    # representative-section virtual limiter sweep.  This produces a blocker map
    # and a final push strategy, so the real render is not a blind list of fixed
    # LUFS attempts.
    virtual_hot_solver = build_virtual_hot_commercial_solver(y, sr, before, decision, mode, base, profile)
    report["virtual_hot_commercial_solver"] = virtual_hot_solver
    report["representative_section_map"] = virtual_hot_solver.get("representative_section_map", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["blocker_capacity_map"] = virtual_hot_solver.get("blocker_capacity_map", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["blocker_solution_plan"] = virtual_hot_solver.get("blocker_solution_plan", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["final_push_strategy"] = virtual_hot_solver.get("final_push_strategy", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["musical_role_identity_map"] = virtual_hot_solver.get("musical_role_identity_map", {}) if isinstance(virtual_hot_solver, dict) else {}
    report["virtual_premaster_strategy"] = {
        "schema_version": "busy_virtual_premaster_strategy_v8_5_3_17",
        "basis": "base analysis / role map / protection map / blocker map",
        "premaster_moves": report.get("blocker_solution_plan", {}),
        "limiting_strategy": report.get("final_push_strategy", {}),
        "policy": "simulate premaster correction candidates first; then build the real premaster once and limit using the selected strategy",
    }

    if target_met and not _env_bool("BUSY_VIRTUAL_HOT_SOLVER_PUSH_WHEN_FLOOR_MET", False):
        report["reason"] = "commercial_floor_already_met"
        report["output"] = report["input"]
        return y, report

    gap_to_floor = max(0.0, floor - base_lufs)
    gap_to_target = max(gap_to_floor, target - base_lufs)
    max_push = float(profile.get("max_push_db", 2.0) or 2.0)
    strategy = virtual_hot_solver.get("final_push_strategy", {}) if isinstance(virtual_hot_solver, dict) and isinstance(virtual_hot_solver.get("final_push_strategy", {}), dict) else {}
    solver_push = float(strategy.get("recommended_input_gain_db", 0.0) or 0.0) if strategy else 0.0
    if solver_push > 0.0:
        # v8.5.3.19: the virtual solver owns the gain ceiling. Do not clamp the
        # real candidate render back to an env/profile max-push if the solver found
        # a higher solvable point.
        max_push = max(max_push, float(solver_push) + 0.35, gap_to_target + 0.25)
    needed = float(np.clip(max(gap_to_target, solver_push), 0.0, max_push))
    if needed < 0.20:
        report["reason"] = "gap_too_small" if not target_met else "commercial_floor_already_met"
        report["output"] = report["input"]
        return y, report

    solver_pushes = strategy.get("recommended_candidate_gains_db", []) if isinstance(strategy.get("recommended_candidate_gains_db", []), list) else []
    pushes = [float(x) for x in solver_pushes if isinstance(x, (int, float)) and float(x) >= 0.18]
    if not pushes:
        pushes = [min(needed, x) for x in (0.55, 0.95, 1.35, 1.80, 2.35, max_push)]
    elif needed not in pushes:
        pushes.append(needed)
    pushes = sorted({round(float(np.clip(x, 0.18, max_push)), 3) for x in pushes if x >= 0.18})
    max_finish_candidates = int(_env_float("BUSY_COMMERCIAL_FINISH_MAX_CANDIDATES", 6 if profile.get("adaptive_hot_commercial") or profile.get("experimental_hot_commercial") else 4, 1, 8))
    if len(pushes) > max_finish_candidates:
        # Keep candidates closest to the solver recommendation, plus the loudest edge.
        focus = solver_push if solver_push > 0 else needed
        pushes = sorted(pushes, key=lambda x: (abs(x - focus), x))[: max_finish_candidates - 1] + [max(pushes)]
        pushes = sorted(set(pushes))

    base_warnings = _safety_risk(before, base, decision)
    base_hard = _commercial_loudness_candidate_hard_blocks(before, base, decision, profile, base_warnings)
    best_audio = y
    best_analysis = base
    best_score = -1e9
    best_name = "base"
    report["retry_attempted"] = True
    report["base_warnings"] = base_warnings
    report["base_hard_blocks"] = base_hard

    for push in pushes:
        try:
            planned_ceiling_db = FINAL_TRUE_PEAK_CEILING_DB
            if isinstance(strategy, dict) and strategy.get("recommended_true_peak_ceiling_db") is not None:
                planned_ceiling_db = float(np.clip(float(strategy.get("recommended_true_peak_ceiling_db")), -2.0, FINAL_TRUE_PEAK_CEILING_DB))
            conditioned, cond_report = _condition_for_commercial_push(y, sr, before, base, push, profile, virtual_hot_plan=virtual_hot_solver)
            cond_report["planned_true_peak_ceiling_db"] = round(float(planned_ceiling_db), 3)
            conditioned_analysis = analyze_audio_fast_qc(conditioned, sr, true_peak_oversample=_qc_oversample_factor())
            cand = apply_gain_db(conditioned, push)
            gain_only_analysis = analyze_audio_fast_qc(cand, sr, true_peak_oversample=_qc_oversample_factor())
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
            cand, hot_render_report = _commercial_hot_candidate_render_with_trace(
                cand,
                sr,
                planned_ceiling_db,
                target,
                floor,
                before,
                decision,
                profile,
            )
            cond_report["candidate_render_trace"] = {
                k: v for k, v in hot_render_report.items() if k != "final_analysis_full"
            }
            tp_gain = float(hot_render_report.get("initial_limiter_true_peak_normalize_gain_db", 0.0) or 0.0)
            cand_analysis = hot_render_report.get("final_analysis_full") if isinstance(hot_render_report.get("final_analysis_full"), dict) else analyze_audio_fast_qc(cand, sr, true_peak_oversample=_qc_oversample_factor())
            warnings = _safety_risk(before, cand_analysis, decision)
            hard_blocks = _commercial_loudness_candidate_hard_blocks(before, cand_analysis, decision, profile, warnings)
            lufs = float(cand_analysis.get("integrated_lufs", -99.0) or -99.0)
            crest = float(cand_analysis.get("crest_factor_db", 0.0) or 0.0)
            reached_floor = bool(lufs >= floor)
            reached_target = bool(lufs >= target - 0.20)
            overshot = bool(lufs > target + 0.75 and crest < float(profile.get("crest_min_db", 5.6) or 5.6) + 0.35)
            hard_blocked = bool(hard_blocks or overshot)
            # Reward floor success and closeness to target, but keep crest/translation
            # ahead of raw loudness when candidates are similar.
            score = 0.0
            score += (lufs - base_lufs) * 9.0
            score -= abs(target - lufs) * 3.2
            score += 18.0 if reached_floor else -18.0
            score += 6.0 if reached_target else 0.0
            score += min(7.0, max(0.0, crest - float(profile.get("hard_crest_min_db", 5.0) or 5.0)) * 1.25)
            score -= len(warnings) * 2.0
            score -= len(hard_blocks) * 24.0
            if overshot:
                score -= 18.0
            rep = {
                "name": f"commercial_push_{push:.2f}db",
                "push_db": round(float(push), 3),
                "true_peak_normalize_gain_db": round(float(tp_gain), 3),
                "conditioning": cond_report,
                "virtual_solver_guided": bool((virtual_hot_solver or {}).get("active")),
                "analysis": _analysis_summary_for_loudness(cand_analysis),
                "warnings": warnings,
                "hard_blocks": hard_blocks + (["commercial_overpush"] if overshot else []),
                "reached_floor": reached_floor,
                "reached_target": reached_target,
                "score": round(float(score), 3),
            }
            report["candidates"].append(rep)
            _debug("commercial_finish_candidate", **{k: v for k, v in rep.items() if k not in {"conditioning", "analysis"}})
            if not hard_blocked and lufs > base_lufs + 0.18 and score > best_score:
                best_audio = cand
                best_analysis = cand_analysis
                best_score = score
                best_name = rep["name"]
            else:
                del cand
        except Exception as exc:
            report["candidates"].append({"name": f"commercial_push_{push:.2f}db", "error": str(exc)[:300]})
            _debug("commercial_finish_candidate_failed", push_db=push, error=str(exc)[:300])

    best_lufs = float(best_analysis.get("integrated_lufs", base_lufs) or base_lufs)
    selected = best_name != "base"
    report["selected"] = best_name
    report["selected_score"] = round(float(best_score), 3) if selected else None
    report["output"] = _analysis_summary_for_loudness(best_analysis)
    report["lufs_gain_achieved_db"] = round(float(best_lufs - base_lufs), 3)
    report["target_met"] = bool(best_lufs >= floor)
    report["target_reached"] = bool(best_lufs >= target - 0.20)
    if report["target_met"]:
        report["reason"] = "commercial_floor_reached_after_repair_then_push" if selected else "commercial_floor_already_met"
    elif selected:
        report["reason"] = "improved_but_commercial_floor_not_reached"
        report["retry_blocked_by"] = "candidate_qc_limits_or_max_push"
    else:
        report["reason"] = "commercial_floor_not_reached"
        blocks: list[str] = []
        for cand in report.get("candidates", []):
            blocks.extend(cand.get("hard_blocks", []) if isinstance(cand, dict) else [])
        report["retry_blocked_by"] = sorted(set(blocks)) or "no_qc_safe_louder_candidate"
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
    _debug("engine_start_fresh_limiter_worker", sr=sr, shape=getattr(pre_limiter, "shape", None), dtype=str(getattr(pre_limiter, "dtype", "")), mode=mode)
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
        pre_limiter,
        sr,
        ceiling_db=FINAL_TRUE_PEAK_CEILING_DB,
        oversample=os_factor,
        lookahead_ms=float(lookahead_ms),
        release_ms=float(release_ms),
    )
    _debug("engine_done_oversampled_final_limiter", output_shape=getattr(final, "shape", None))

    _debug("engine_start_true_peak_normalize", oversample=os_factor, target_db=FINAL_TRUE_PEAK_CEILING_DB)
    final, final_true_peak_gain_db = _normalize_to_true_peak(final, FINAL_TRUE_PEAK_CEILING_DB, sr=sr, oversample=os_factor)
    _debug("engine_done_true_peak_normalize", gain_db=round(final_true_peak_gain_db, 3))

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
    commercial_source_for_finish = pre_limiter if "pre_limiter" in locals() else final
    commercial_candidate, commercial_finish_report = _commercial_loudness_finish_pass(commercial_source_for_finish, sr, before, decision, mode)
    commercial_selected = bool(commercial_finish_report.get("selected") and commercial_finish_report.get("selected") != "base")
    commercial_finish_report["strategy_application_stage"] = "pre_limiter_before_final_safety_limiter"
    # v8.5.3.20: never replace the already rendered standard final with a
    # commercial candidate that is quieter.  The finish pass is evaluated against
    # the pre-limiter source, so its internal base can be much quieter than the
    # standard final.  This guard prevents a "selected" pre-limiter candidate from
    # making the delivered master smaller.
    standard_final_reference = analyze_audio_fast_qc(final, sr, true_peak_oversample=_qc_oversample_factor())
    commercial_finish_report["standard_final_reference_analysis"] = _analysis_summary_for_loudness(standard_final_reference)
    try:
        std_lufs = float(standard_final_reference.get("integrated_lufs", -99.0) or -99.0)
        cand_lufs = float((commercial_finish_report.get("output") or {}).get("integrated_lufs", -99.0) or -99.0)
    except Exception:
        std_lufs, cand_lufs = -99.0, -99.0
    if commercial_selected and cand_lufs < std_lufs + 0.12 and not bool(commercial_finish_report.get("target_met")):
        commercial_finish_report["candidate_rejected_against_standard_final"] = True
        commercial_finish_report["candidate_rejection_reason"] = "candidate_not_louder_than_standard_final"
        commercial_finish_report["candidate_vs_standard_lufs_delta_db"] = round(float(cand_lufs - std_lufs), 3)
        commercial_selected = False
    commercial_finish_report["selected_candidate_replaced_standard_final_limiter"] = bool(commercial_selected)
    if commercial_selected:
        final = commercial_candidate
    else:
        commercial_finish_report["fallback_final_source"] = "standard_final_limiter_output"
    _debug(
        "engine_done_commercial_loudness_finish",
        active=commercial_finish_report.get("active"),
        target_met=commercial_finish_report.get("target_met"),
        selected=commercial_finish_report.get("selected"),
        reason=commercial_finish_report.get("reason"),
        output=(commercial_finish_report.get("output") or {}).get("integrated_lufs"),
        strategy_application_stage=commercial_finish_report.get("strategy_application_stage"),
    )

    final, dither_report = _apply_output_dither(final, bit_depth=24)
    if dither_report.get("active"):
        _debug("engine_done_dither_noise_shaping", **dither_report)

    _debug("engine_start_final_analysis")
    final_analysis = analyze_audio(final, sr)
    reasons = _safety_risk(before, final_analysis, decision)
    stereo_qc = _stereo_qc_summary(final_analysis)
    ab_delta_qc = _ab_delta_qc(before, final_analysis)
    lra_guard = _lra_guard_summary(before, final_analysis, decision, mode)
    codec_preview_guard = _codec_preview_guard(before, final_analysis)
    playback_translation = _playback_translation_report(before, final_analysis)
    reasons = sorted(set(
        reasons
        + stereo_qc.get("warnings", [])
        + ab_delta_qc.get("warnings", [])
        + lra_guard.get("warnings", [])
        + codec_preview_guard.get("warnings", [])
        + playback_translation.get("warnings", [])
    ))
    _debug("engine_done_stereo_qc", **stereo_qc)
    _debug("engine_done_lra_guard", **lra_guard)
    _debug("engine_done_codec_preview_guard", **codec_preview_guard)
    _debug("engine_done_playback_translation", overall_score=playback_translation.get("overall_score"), warnings=playback_translation.get("warnings", []))
    _debug("engine_done_ab_delta_qc", warnings=ab_delta_qc.get("warnings", []), deltas=ab_delta_qc.get("deltas", {}))
    _debug("engine_done_final_analysis", reasons=reasons, lufs=final_analysis.get("integrated_lufs"), lra=final_analysis.get("lra_lu"), true_peak=final_analysis.get("approx_true_peak_dbfs"), crest=final_analysis.get("crest_factor_db"), correlation=final_analysis.get("correlation"))

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
        "schema_version": "busy_actual_vs_predicted_qc_v8_5_3_20",
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
        "actual_vs_predicted_qc": actual_vs_predicted_qc,
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
        "codec_preview_guard": codec_preview_guard,
        "playback_translation_auditor": playback_translation_auditor,
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

    report = {
        "engine_version": "busy_auto_mastering_v8_5_3_20_candidate_limiter_trace_fix",
        "mastering_report_schema_version": "busy_master_report_v8_5_3_20_candidate_limiter_trace_fix",
        "two_stage_limiting": True,
        "input_analysis": before,
        "original_input_analysis_before_pre_clean": pre_limiter_report.get("original_input_analysis_before_pre_clean", {}),
        "residue_pre_clean_stage": pre_limiter_report.get("residue_pre_clean_stage", {}),
        "pre_limiter_report": pre_limiter_report,
        "output_analysis": final_analysis,
        "working_stage_gain_db": pre_limiter_report.get("working_stage_gain_db"),
        "final_true_peak_gain_db": round(final_true_peak_gain_db, 3),
        "final_true_peak_target_dbtp": FINAL_TRUE_PEAK_CEILING_DB,
        "requested_oversample_factor": adaptive_quality_report.get("requested_oversample"),
        "oversample_factor": os_factor,
        "adaptive_quality": adaptive_quality_report,
        "loudness_push_governor": pre_limiter_report.get("loudness_push_governor"),
        "stereo_qc": stereo_qc,
        "ab_delta_qc": ab_delta_qc,
        "lra_guard": lra_guard,
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
        "virtual_hot_commercial_solver": commercial_finish_report.get("virtual_hot_commercial_solver", {}),
        "representative_section_map": commercial_finish_report.get("representative_section_map", {}),
        "virtual_limiter_sweep": (commercial_finish_report.get("virtual_hot_commercial_solver", {}) or {}).get("virtual_limiter_sweep", {}) if isinstance(commercial_finish_report.get("virtual_hot_commercial_solver", {}), dict) else {},
        "blocker_capacity_map": commercial_finish_report.get("blocker_capacity_map", {}),
        "blocker_solution_plan": commercial_finish_report.get("blocker_solution_plan", {}),
        "final_push_strategy": commercial_finish_report.get("final_push_strategy", {}),
        "musical_role_identity_map": commercial_finish_report.get("musical_role_identity_map", {}),
        "virtual_premaster_strategy": commercial_finish_report.get("virtual_premaster_strategy", {}),
        "actual_vs_predicted_qc": actual_vs_predicted_qc,
        "frequency_band_centrifuge": before.get("frequency_band_centrifuge", {}) if isinstance(before, dict) else {},
        "commercial_loudness_result": {
            "target_lufs": commercial_finish_report.get("target_lufs"),
            "commercial_floor_lufs": commercial_finish_report.get("commercial_floor_lufs"),
            "final_lufs": final_analysis.get("integrated_lufs"),
            "target_met": commercial_finish_report.get("target_met"),
            "target_reached": commercial_finish_report.get("target_reached"),
            "retry_attempted": commercial_finish_report.get("retry_attempted"),
            "reason": commercial_finish_report.get("reason"),
            "retry_blocked_by": commercial_finish_report.get("retry_blocked_by"),
            "policy": "commercial loudness floor is checked before accepting a quiet master",
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
    _debug("engine_done", output_lufs=final_analysis.get("integrated_lufs"), output_true_peak=final_analysis.get("approx_true_peak_dbfs"), oversample_factor=os_factor, adaptive_quality=adaptive_quality_report)
    return np.nan_to_num(final, nan=0.0, posinf=0.0, neginf=0.0), report
