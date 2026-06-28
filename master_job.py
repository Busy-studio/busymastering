from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
import resource
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ai_decision import build_mastering_decision
from ai_decision.genre_evidence import build_genre_evidence
from ai_decision.realtime_audio_judge import run_realtime_audio_genre_judge, run_realtime_audio_decision_validator
from ai_decision.gemini_audio_judge import run_gemini_audio_genre_judge, run_gemini_audio_decision_validator
from ai_decision.private_rules import load_private_bundle, load_private_rulepack
from ai_decision.reference_v7 import build_v7_overlay, is_v7_db
from ai_decision.role_dsp_contract import apply_role_first_dsp_contract
from audio_engine import analyze_audio, analyze_segments, analyze_suno_artifacts, load_audio_file, master_audio, preflight_clean_audio
from audio_engine.busy_auto_mixing import build_busy_auto_mixing_premaster
from audio_engine.essentia_features import extract_essentia_evidence
from audio_engine.genre_preview import build_realtime_genre_preview
from audio_engine.adaptive_preview import build_adaptive_audio_preview_plan
from audio_engine.mix_conditioning import (
    build_mix_conditioning_plan,
    apply_mix_conditioning_to_decision,
    build_loudness_feasibility_score,
    calculate_perceptual_quality_score,
)
from audio_engine.research_capability_framework import (
    apply_research_capability_framework,
    attach_research_capability_to_report,
    compact_research_capability_summary,
    ENGINE_VERSION as RESEARCH_CAPABILITY_ENGINE_VERSION,
    REPORT_SCHEMA as RESEARCH_CAPABILITY_REPORT_SCHEMA,
    SCHEMA_VERSION as RESEARCH_CAPABILITY_SCHEMA_VERSION,
)
from audio_engine.multicomponent_assist import analyze_multicomponent_assist
from audio_engine.stem_evidence_assist import analyze_stem_evidence_assist, build_stem_evidence_pack
from audio_engine.stem_sum_residue_witness import compact_ssrw_summary
from audio_engine.io import write_wav_pcm24
from audio_engine.mastering import render_pre_limiter_master, finalize_limiter_master, prepare_pre_limiter_stage, render_pre_limiter_from_prepared_stage, _v6314_4_finalize_floor_tolerance_and_handoff_markers, _v6362_8_finalize_report_authority
from audio_engine.finishing_intelligence import attach_finishing_intelligence_to_report, ENGINE_VERSION as V6400_ENGINE_VERSION, PIPELINE_VERSION as V6400_PIPELINE_VERSION, REPORT_SCHEMA as V6400_REPORT_SCHEMA
from audio_engine.auto_intensity import (
    normalize_mastering_intensity as _auto_normalize_mastering_intensity,
    apply_mastering_intensity_env_to_process as _auto_apply_mastering_intensity_env_to_process,
    select_auto_mastering_intensity,
)
from audio_engine.residue_centrifuge import analyze_frequency_band_centrifuge
from audio_engine.pipeline_architecture import (
    build_pre_clean_plan,
    build_post_pre_clean_analysis,
    build_blocker_remediation_plan,
    build_pipeline_architecture_state,
    render_budget_policy,
)
from supabase_storage import SupabaseStorage



def _normalize_mastering_intensity(value: Any) -> str:
    return _auto_normalize_mastering_intensity(value, default=os.environ.get("BUSY_MASTERING_INTENSITY", "auto"), allow_auto=True)


def _normalize_final_mastering_intensity(value: Any) -> str:
    return _auto_normalize_mastering_intensity(value, default="balanced", allow_auto=False)


def _apply_mastering_intensity_env_to_process(intensity: str, policy: dict[str, Any] | None = None) -> str:
    return _auto_apply_mastering_intensity_env_to_process(intensity, policy=policy)


def _effective_mastering_intensity(decision: dict[str, Any] | None = None, requested: Any = None) -> str:
    d = decision if isinstance(decision, dict) else {}
    auto_policy = d.get("auto_intensity_policy") if isinstance(d.get("auto_intensity_policy"), dict) else {}
    return _normalize_final_mastering_intensity(auto_policy.get("final_mode") or d.get("mastering_intensity") or requested or os.environ.get("BUSY_MASTERING_INTENSITY") or "balanced")


def _resolve_auto_intensity_policy(features: dict[str, Any], decision: dict[str, Any], requested: Any, log_fp, phase: str = "A-1") -> dict[str, Any]:
    policy = select_auto_mastering_intensity(features, decision, requested_mode=requested)
    final = _apply_mastering_intensity_env_to_process(policy.get("final_mode"), policy=policy)
    policy["env_applied"] = {
        "BUSY_MASTERING_INTENSITY": os.environ.get("BUSY_MASTERING_INTENSITY"),
        "BUSY_TRUE_PEAK_CEILING_DB": os.environ.get("BUSY_TRUE_PEAK_CEILING_DB"),
        "BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB": os.environ.get("BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB"),
        "BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB": os.environ.get("BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB"),
        "BUSY_AUTO_INTENSITY_CHASE_STRENGTH": os.environ.get("BUSY_AUTO_INTENSITY_CHASE_STRENGTH"),
    }
    features["auto_intensity_policy"] = policy
    decision["auto_intensity_policy"] = policy
    decision["mastering_intensity"] = final
    decision["mastering_intensity_requested"] = _normalize_mastering_intensity(requested)
    flags = list(decision.get("processing_flags") or []) if isinstance(decision.get("processing_flags"), list) else []
    if "auto_mastering_intensity_policy_active" not in flags:
        flags.append("auto_mastering_intensity_policy_active")
    decision["processing_flags"] = flags
    _log(
        log_fp,
        "stage_done_auto_mastering_intensity_policy",
        phase=phase,
        requested=policy.get("requested_mode"),
        prior=policy.get("prior_mode"),
        final=policy.get("final_mode"),
        family=policy.get("family_bucket"),
        risk_score=policy.get("risk_score"),
        hard_veto_flags=policy.get("hard_veto_flags"),
        protective_caps=policy.get("protective_caps"),
        tp_ceiling=policy.get("selected_true_peak_ceiling_dbtp"),
        chase_strength=policy.get("selected_chase_strength"),
    )
    return policy


def _apply_intensity_from_stage_state(state_in: dict[str, Any], args: argparse.Namespace | None, log_fp, phase: str = "state_load") -> str:
    decision = state_in.get("decision", {}) if isinstance(state_in, dict) else {}
    requested = getattr(args, "mastering_intensity", None) if args is not None else None
    policy = decision.get("auto_intensity_policy") if isinstance(decision, dict) and isinstance(decision.get("auto_intensity_policy"), dict) else {}
    if not policy and isinstance(state_in.get("auto_intensity_policy"), dict):
        policy = state_in.get("auto_intensity_policy") or {}
    final = policy.get("final_mode")
    final = final or state_in.get("mastering_intensity") or (decision.get("mastering_intensity") if isinstance(decision, dict) else None) or requested or "balanced"
    final = _apply_mastering_intensity_env_to_process(final, policy=policy if policy else None)
    if args is not None:
        args.mastering_intensity = final
    try:
        state_in["mastering_intensity"] = final
        if isinstance(decision, dict):
            decision["mastering_intensity"] = final
            # v8.5.3.33: keep the A-1 auto-intensity decision as the single
            # source of truth for every downstream stage.  Some split-stage
            # workers receive --mastering-intensity auto on their CLI; without
            # reattaching the policy, the process can silently fall back to
            # Balanced while the stage_state still says Safe.
            if policy:
                decision["auto_intensity_policy"] = policy
                decision["mastering_intensity_locked"] = True
    except Exception:
        pass
    _log(log_fp, "stage_done_apply_mastering_intensity_from_state", phase=phase, final=final, requested=requested, has_auto_policy=bool(policy), true_peak_ceiling=os.environ.get("BUSY_TRUE_PEAK_CEILING_DB"), commercial_reference_true_peak=os.environ.get("BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB"), chase_strength=os.environ.get("BUSY_AUTO_INTENSITY_CHASE_STRENGTH"))
    return final


def _mode_rank(mode: Any) -> int:
    order = ["safe", "balanced", "commercial", "maximum"]
    try:
        return order.index(_normalize_final_mastering_intensity(mode))
    except Exception:
        return 1


def _bamix_memory_handoff_barrier(log_fp, phase: str = "bamix_to_mastering_handoff") -> dict[str, Any]:
    before = _rss_mb()
    malloc_trim_result = None
    errors: list[str] = []
    try:
        gc.collect()
        gc.collect()
    except Exception as exc:
        errors.append(f"gc:{type(exc).__name__}")
    if _env_on("BUSY_AUTOMIX_MALLOC_TRIM", "1"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            malloc_trim_result = int(libc.malloc_trim(0))
        except Exception as exc:
            errors.append(f"malloc_trim:{type(exc).__name__}")
    after = _rss_mb()
    payload = {
        "schema_version": "busy_auto_mixing_memory_handoff_v8_5_3_63_1",
        "phase": phase,
        "rss_before_mb": before,
        "rss_after_mb": after,
        "malloc_trim_result": malloc_trim_result,
        "errors": errors,
        "policy": "release BAMix/original/stem buffers before downstream mastering loads the premaster",
    }
    _log(log_fp, "stage_done_busy_auto_mixing_memory_handoff_barrier", **payload)
    return payload


def _clear_bamix_runtime_handoff_env() -> None:
    """Clear run-specific BAMix handoff env left in warm Cloud Run containers.

    Keep user/config tuning envs intact; only remove values that are written from
    a previous job's BAMix policy.  Current-job policy is still stored in the
    decision/report and will be re-exported when BAMix actually succeeds.
    """
    for key in (
        "BUSY_AUTOMIX_PREMASTER_ACTIVE",
        "BUSY_BAMIX_V6319_HANDOFF_MODE",
        "BUSY_AUTOMIX_MASTERING_EFFECTIVE_TARGET_LUFS",
        "BUSY_AUTOMIX_MASTERING_EFFECTIVE_FLOOR_LUFS",
        "BUSY_AUTOMIX_MASTERING_MAX_CREST_LOSS_DB",
        "BUSY_AUTOMIX_MASTERING_PREMASTER_LRA_LU",
        "BUSY_AUTOMIX_MASTERING_PREMASTER_CREST_DB",
        "BUSY_BAMIX_SELECTED_RECIPE",
        "BUSY_BAMIX_PUNCH_PRESERVED",
    ):
        try:
            os.environ.pop(key, None)
        except Exception:
            pass


def _bamix_v6319_num(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
        if np.isfinite(out):
            return float(out)
    except Exception:
        pass
    return default


def _derive_bamix_v6319_commercial_handoff(
    bamix_report: dict[str, Any],
    *,
    current_mode: str,
    max_intensity: str,
    capped: bool,
) -> dict[str, Any]:
    """Derive a BAMix -> mastering governance policy.

    v63.1.9 changes the meaning of BAMix handoff from a hard transparent-polish
    brake into a readiness-aware push governor.  BAMix still remains the source
    authority, but downstream mastering may complete commercial density when the
    premaster is structurally healthy and under-dense.
    """
    premaster_qc = bamix_report.get("premaster_qc") if isinstance(bamix_report.get("premaster_qc"), dict) else {}
    recipe = str(bamix_report.get("selected_mix_recipe") or bamix_report.get("recipe") or "").strip().lower()
    punch_preserved = recipe == "punch_preserved"
    lufs = _bamix_v6319_num(premaster_qc.get("integrated_lufs"), None)
    crest = _bamix_v6319_num(premaster_qc.get("crest_factor_db"), None)
    lra = _bamix_v6319_num(premaster_qc.get("lra_lu"), None)
    corr = _bamix_v6319_num(premaster_qc.get("phase_correlation"), None)
    tp = _bamix_v6319_num(premaster_qc.get("true_peak_dbtp"), None)
    hard_warnings = [str(x) for x in (premaster_qc.get("hard_warnings") or []) if x]
    soft_warnings = [str(x) for x in (premaster_qc.get("soft_warnings") or []) if x]
    all_warnings = [str(x) for x in (premaster_qc.get("warnings") or hard_warnings + soft_warnings) if x]
    _targets = premaster_qc.get("premaster_targets") if isinstance(premaster_qc.get("premaster_targets"), dict) else {}
    v6340_contract = _targets.get("v6341_reference_db_handoff_contract") if isinstance(_targets.get("v6341_reference_db_handoff_contract"), dict) else (_targets.get("v6340_premaster_handoff_contract") if isinstance(_targets.get("v6340_premaster_handoff_contract"), dict) else {})
    v6340_contract_active = bool(v6340_contract.get("active")) and _env_on("BUSY_BAMIX_V6340_HANDOFF_CONTRACT", "1")
    v6350_workload_plan = premaster_qc.get("v6350_density_limiter_workload_plan") if isinstance(premaster_qc.get("v6350_density_limiter_workload_plan"), dict) else {}
    v6340_budget = v6340_contract.get("final_limiter_gain_budget_lu") if isinstance(v6340_contract.get("final_limiter_gain_budget_lu"), (list, tuple)) else []
    try:
        v6340_budget_min = float(v6340_budget[0]) if len(v6340_budget) >= 1 else None
        v6340_budget_target = float(v6340_budget[1]) if len(v6340_budget) >= 2 else None
        v6340_budget_max = float(v6340_budget[2]) if len(v6340_budget) >= 3 else None
    except Exception:
        v6340_budget_min = v6340_budget_target = v6340_budget_max = None

    stem_metrics = bamix_report.get("stem_metrics") if isinstance(bamix_report.get("stem_metrics"), list) else []
    art_values = []
    conf_values = []
    for item in stem_metrics:
        if not isinstance(item, dict):
            continue
        av = _bamix_v6319_num(item.get("artifact_risk"), None)
        cv = _bamix_v6319_num(item.get("role_confidence"), None)
        if av is not None:
            art_values.append(float(av))
        if cv is not None:
            conf_values.append(float(cv))
    max_art = max(art_values) if art_values else 0.0
    mean_art = float(np.mean(art_values)) if art_values else 0.0
    mean_conf = float(np.mean(conf_values)) if conf_values else None

    commercial_target = _env_float("BUSY_BAMIX_V632_COMMERCIAL_TARGET_LUFS", _env_float("BUSY_BAMIX_V6319_COMMERCIAL_TARGET_LUFS", -9.25, -12.5, -8.0), -12.5, -8.0)
    transparent_target = _env_float("BUSY_BAMIX_V6319_TRANSPARENT_TARGET_LUFS", -11.5, -14.0, -9.5)
    target_gap = max(0.0, float(commercial_target) - float(lufs)) if lufs is not None else None

    readiness_score = 100.0
    readiness_reasons: list[str] = []
    if bool(premaster_qc.get("hard_fail")):
        readiness_score -= 28.0
        readiness_reasons.append("premaster_qc_hard_fail_marker")
    if max_art >= 0.85:
        readiness_score -= 24.0
        readiness_reasons.append("severe_stem_artifact_risk")
    elif max_art >= 0.68:
        readiness_score -= 13.0
        readiness_reasons.append("elevated_stem_artifact_risk")
    if mean_art >= 0.50:
        readiness_score -= 8.0
        readiness_reasons.append("mean_artifact_risk_elevated")
    if mean_conf is not None and mean_conf < 0.55:
        readiness_score -= 8.0
        readiness_reasons.append("low_role_confidence")
    if corr is not None and corr < _env_float("BUSY_BAMIX_V6319_FRAGILE_CORRELATION", 0.18, -0.20, 0.80):
        readiness_score -= 16.0
        readiness_reasons.append("phase_correlation_fragile")
    if lra is not None and lra < _env_float("BUSY_BAMIX_V6319_FRAGILE_LRA_LU", 2.65, 0.5, 6.0):
        readiness_score -= 17.0
        readiness_reasons.append("premaster_lra_fragile")
    if crest is not None and crest < _env_float("BUSY_BAMIX_V6319_FRAGILE_CREST_DB", 9.0, 4.0, 14.0):
        readiness_score -= 14.0
        readiness_reasons.append("premaster_crest_fragile")
    if tp is not None and tp > _env_float("BUSY_BAMIX_V6319_TP_NEAR_LIMIT_DBTP", -0.7, -3.0, 0.5):
        readiness_score -= 8.0
        readiness_reasons.append("limited_true_peak_headroom")
    if any("phase" in w for w in all_warnings):
        readiness_score -= 10.0
        readiness_reasons.append("premaster_phase_warning")
    if any("underdriven" in w or "high_crest_low_density" in w for w in all_warnings):
        readiness_reasons.append("premaster_density_completion_needed")

    readiness_score = float(np.clip(readiness_score, 0.0, 100.0))
    if readiness_score >= 76.0:
        readiness = "high"
    elif readiness_score >= 58.0:
        readiness = "medium"
    elif readiness_score >= 40.0:
        readiness = "low"
    else:
        readiness = "fragile"

    underdense_gap = bool(target_gap is not None and target_gap >= _env_float("BUSY_BAMIX_V632_DENSITY_GAP_LU", _env_float("BUSY_BAMIX_V6319_DENSITY_GAP_LU", 1.05, 0.2, 4.0), 0.2, 5.0))
    big_gap = bool(target_gap is not None and target_gap >= _env_float("BUSY_BAMIX_V632_BIG_DENSITY_GAP_LU", _env_float("BUSY_BAMIX_V6319_BIG_DENSITY_GAP_LU", 2.25, 0.8, 6.0), 0.8, 7.5))
    fragile = bool(readiness == "fragile" or (lra is not None and lra < 2.25) or (crest is not None and crest < 8.5) or max_art >= 0.90 or (corr is not None and corr < 0.10))

    if fragile:
        handoff_mode = "fragile_protection"
    elif underdense_gap and readiness in {"high", "medium"}:
        handoff_mode = "commercial_density_completion"
    elif underdense_gap:
        handoff_mode = "guarded_completion"
    else:
        handoff_mode = "transparent_polish"

    # Mode-dependent push budgets. Existing v57/v60/v61/export QC still decide
    # whether a candidate is safe; these numbers only describe the allowed route.
    if handoff_mode == "commercial_density_completion":
        max_total_gain = _env_float("BUSY_BAMIX_V632_DENSITY_MAX_TOTAL_GAIN_LU", _env_float("BUSY_BAMIX_V6319_DENSITY_MAX_TOTAL_GAIN_LU", 6.9 if not big_gap else 7.8, 3.0, 10.5), 3.0, 10.5)
        max_finish_push = _env_float("BUSY_BAMIX_V632_DENSITY_FINISH_PUSH_DB", _env_float("BUSY_BAMIX_V6319_DENSITY_FINISH_PUSH_DB", 3.55, 0.6, 5.2), 0.6, 5.2)
        chase = _env_float("BUSY_BAMIX_V632_DENSITY_CHASE_STRENGTH", _env_float("BUSY_BAMIX_V6319_DENSITY_CHASE_STRENGTH", 0.62, 0.0, 0.95), 0.0, 0.95)
        max_crest_loss = _env_float("BUSY_BAMIX_V632_DENSITY_MAX_CREST_LOSS_DB", _env_float("BUSY_BAMIX_V6319_DENSITY_MAX_CREST_LOSS_DB", 5.35, 1.5, 7.4), 1.5, 7.4)
        floor_gap = _env_float("BUSY_BAMIX_V632_DENSITY_FLOOR_GAP_DB", _env_float("BUSY_BAMIX_V6319_DENSITY_FLOOR_GAP_DB", 0.75, 0.3, 2.0), 0.3, 2.0)
        preferred_target = commercial_target
        profile = "bamix_commercial_density_completion"
        prefer_cleaner = False
    elif handoff_mode == "guarded_completion":
        max_total_gain = _env_float("BUSY_BAMIX_V6319_GUARDED_MAX_TOTAL_GAIN_LU", 5.3, 2.0, 8.0)
        max_finish_push = _env_float("BUSY_BAMIX_V6319_GUARDED_FINISH_PUSH_DB", 2.25, 0.4, 4.0)
        chase = _env_float("BUSY_BAMIX_V6319_GUARDED_CHASE_STRENGTH", 0.38, 0.0, 0.8)
        max_crest_loss = _env_float("BUSY_BAMIX_V6319_GUARDED_MAX_CREST_LOSS_DB", 3.8, 1.0, 6.0)
        floor_gap = _env_float("BUSY_BAMIX_V6319_GUARDED_FLOOR_GAP_DB", 0.95, 0.3, 2.2)
        preferred_target = min(float(commercial_target), _env_float("BUSY_BAMIX_V6319_GUARDED_HOTTEST_TARGET_LUFS", -10.1, -12.5, -8.5))
        profile = "bamix_guarded_density_completion"
        prefer_cleaner = False
    elif handoff_mode == "fragile_protection":
        max_total_gain = _env_float("BUSY_BAMIX_V6319_FRAGILE_MAX_TOTAL_GAIN_LU", 3.8, 1.0, 7.0)
        max_finish_push = _env_float("BUSY_BAMIX_V6319_FRAGILE_FINISH_PUSH_DB", 0.95, 0.0, 2.5)
        chase = _env_float("BUSY_BAMIX_V6319_FRAGILE_CHASE_STRENGTH", 0.18, 0.0, 0.6)
        max_crest_loss = _env_float("BUSY_BAMIX_V6319_FRAGILE_MAX_CREST_LOSS_DB", 2.65, 0.8, 4.5)
        floor_gap = _env_float("BUSY_BAMIX_V6319_FRAGILE_FLOOR_GAP_DB", 0.75, 0.3, 2.0)
        preferred_target = transparent_target
        profile = "bamix_fragile_transparent_protection"
        prefer_cleaner = True
    else:
        max_total_gain = _env_float("BUSY_BAMIX_V6319_TRANSPARENT_MAX_TOTAL_GAIN_LU", 4.6, 1.0, 8.0)
        max_finish_push = _env_float("BUSY_BAMIX_V6319_TRANSPARENT_FINISH_PUSH_DB", 1.25, 0.0, 3.0)
        chase = _env_float("BUSY_BAMIX_V6319_TRANSPARENT_CHASE_STRENGTH", 0.24, 0.0, 0.8)
        max_crest_loss = _env_float("BUSY_BAMIX_V6319_TRANSPARENT_MAX_CREST_LOSS_DB", 3.05, 1.0, 5.0)
        floor_gap = _env_float("BUSY_BAMIX_V6319_TRANSPARENT_FLOOR_GAP_DB", 0.7, 0.3, 2.0)
        preferred_target = transparent_target
        profile = "bamix_transparent_polish"
        prefer_cleaner = True

    if punch_preserved and handoff_mode in {"transparent_polish", "fragile_protection"}:
        max_finish_push = min(max_finish_push, _env_float("BUSY_BAMIX_V6319_PUNCH_TRANSPARENT_FINISH_PUSH_CAP_DB", 1.35, 0.2, 3.0))
        max_crest_loss = min(max_crest_loss, _env_float("BUSY_BAMIX_V6319_PUNCH_TRANSPARENT_MAX_CREST_LOSS_DB", 3.05, 1.0, 5.0))
    elif punch_preserved and handoff_mode in {"commercial_density_completion", "guarded_completion"}:
        readiness_reasons.append("punch_preserved_allowed_density_completion_with_existing_damage_gates")

    if v6340_contract_active and v6340_budget_max is not None:
        # v63.4.0: the final limiter has a workload budget.  If the premaster
        # cannot reach the final target inside that budget, cap the effective
        # target and force the next run/patch to solve density upstream instead
        # of letting the limiter rescue +5~7 LU.
        budget_margin = _env_float("BUSY_BAMIX_V6340_LIMITER_BUDGET_MARGIN_LU", 0.35, 0.0, 1.5)
        max_total_gain = min(float(max_total_gain), float(v6340_budget_max) + float(budget_margin))
        max_finish_push = min(float(max_finish_push), float(v6340_budget_max))
        if v6340_budget_target is not None:
            chase = min(float(chase), _env_float("BUSY_BAMIX_V6340_MAX_CHASE_STRENGTH", 0.58 if handoff_mode in {"commercial_density_completion", "guarded_completion"} else 0.34, 0.0, 0.95))
        max_crest_loss = min(float(max_crest_loss), _env_float("BUSY_BAMIX_V6340_MAX_CREST_LOSS_DB", 4.6 if handoff_mode in {"commercial_density_completion", "guarded_completion"} else 3.2, 1.0, 7.0))
        readiness_reasons.append("v6341_reference_db_or_contract_final_limiter_workload_budget_active")

    effective_target = float(preferred_target)
    if lufs is not None:
        hottest_allowed_by_total_gain = float(lufs) + float(max_total_gain)
        if effective_target > hottest_allowed_by_total_gain:
            effective_target = hottest_allowed_by_total_gain
            readiness_reasons.append("target_capped_by_mode_total_gain_budget")
    hottest_abs = _env_float("BUSY_BAMIX_V632_HOTTEST_EFFECTIVE_TARGET_LUFS", _env_float("BUSY_BAMIX_V6319_HOTTEST_EFFECTIVE_TARGET_LUFS", -8.45, -10.5, -7.6), -10.5, -7.6)
    if effective_target > hottest_abs:
        effective_target = hottest_abs
        readiness_reasons.append("target_capped_by_global_hottest_bamix_target")
    effective_floor = float(effective_target) - float(floor_gap)

    return {
        "schema_version": "busy_auto_mixing_mastering_handoff_policy_v8_5_3_63_5_0_1_density_limiter_workload_hotfix",
        "active": True,
        "profile": profile,
        "handoff_mode": handoff_mode,
        "commercial_readiness": {
            "score": round(float(readiness_score), 3),
            "label": readiness,
            "target_gap_lu": round(float(target_gap), 3) if target_gap is not None else None,
            "underdense_gap": bool(underdense_gap),
            "big_density_gap": bool(big_gap),
            "reasons": readiness_reasons,
            "mean_stem_artifact_risk": round(float(mean_art), 4),
            "max_stem_artifact_risk": round(float(max_art), 4),
            "mean_role_confidence": round(float(mean_conf), 4) if mean_conf is not None else None,
        },
        "current_mode_before_cap": current_mode,
        "max_intensity_for_bamix_premaster": max_intensity,
        "final_mode_after_cap": current_mode if not capped else max_intensity,
        "capped": bool(capped),
        "premaster_lufs": round(float(lufs), 3) if lufs is not None else None,
        "premaster_crest_factor_db": round(float(crest), 3) if crest is not None else None,
        "premaster_lra_lu": round(float(lra), 3) if lra is not None else None,
        "premaster_true_peak_dbtp": round(float(tp), 3) if tp is not None else None,
        "premaster_phase_correlation": round(float(corr), 4) if corr is not None else None,
        "selected_mix_recipe": recipe or None,
        "v632_commercial_premaster_density": {
            "active": True,
            "goal": "shift work from final limiter rescue to BAMix bus-level premaster density",
            "commercial_target_lufs": round(float(commercial_target), 3),
            "target_gap_lu": round(float(target_gap), 3) if target_gap is not None else None,
        },
        "v6350_density_limiter_workload_architecture": {
            "active": bool(isinstance(v6350_workload_plan, dict) and v6350_workload_plan.get("enabled", True)),
            "router_active": bool(isinstance(v6350_workload_plan, dict) and v6350_workload_plan.get("active")),
            "estimated_final_limiter_workload_lu": v6350_workload_plan.get("estimated_final_limiter_workload_lu") if isinstance(v6350_workload_plan, dict) else None,
            "workload_excess_over_target_lu": v6350_workload_plan.get("workload_excess_over_target_lu") if isinstance(v6350_workload_plan, dict) else None,
            "workload_excess_over_max_lu": v6350_workload_plan.get("workload_excess_over_max_lu") if isinstance(v6350_workload_plan, dict) else None,
            "recommended_density_drive_db": v6350_workload_plan.get("recommended_density_drive_db") if isinstance(v6350_workload_plan, dict) else None,
            "recommended_tp_safe_makeup_db": v6350_workload_plan.get("recommended_tp_safe_makeup_db") if isinstance(v6350_workload_plan, dict) else None,
            "reasons": v6350_workload_plan.get("reasons") if isinstance(v6350_workload_plan, dict) else [],
            "policy": "priority-1 handoff: excess final limiter workload must be handled upstream by density/clipper/parallel RMS before final limiter budget is spent.",
        },
        "v6341_reference_db_handoff_contract": {
            "active": bool(v6340_contract_active),
            "source": v6340_contract.get("source"),
            "mode": v6340_contract.get("mode"),
            "primary_reference_profile_id": v6340_contract.get("primary_reference_profile_id"),
            "reference_profile_ids": v6340_contract.get("reference_profile_ids"),
            "profile": v6340_contract.get("profile"),
            "genre_bucket": v6340_contract.get("genre_bucket"),
            "delivery_target_lufs": v6340_contract.get("delivery_target_lufs"),
            "premaster_lufs_min": v6340_contract.get("premaster_lufs_min"),
            "premaster_lufs_target": v6340_contract.get("premaster_lufs_target"),
            "premaster_lufs_max": v6340_contract.get("premaster_lufs_max"),
            "premaster_tp_target_dbtp": v6340_contract.get("premaster_tp_target_dbtp"),
            "premaster_tp_max_dbtp": v6340_contract.get("premaster_tp_max_dbtp"),
            "final_limiter_gain_budget_lu": list(v6340_budget) if isinstance(v6340_budget, (list, tuple)) else None,
            "clipping_limiter_profile": v6340_contract.get("clipping_limiter_profile") if isinstance(v6340_contract.get("clipping_limiter_profile"), dict) else None,
            "budget_cap_applied_to_mastering_policy": bool(v6340_contract_active and v6340_budget_max is not None),
        },
        "punch_preserved": bool(punch_preserved),
        "preferred_final_target_lufs": round(float(preferred_target), 3),
        "effective_final_target_lufs": round(float(effective_target), 3),
        "effective_floor_lufs": round(float(effective_floor), 3),
        "floor_gap_db": round(float(floor_gap), 3),
        "max_total_gain_lufs": round(float(max_total_gain), 3),
        "max_finish_push_db": round(float(max_finish_push), 3),
        "chase_strength": round(float(chase), 3),
        "true_peak_ceiling_dbtp": round(float(_env_float("BUSY_AUTOMIX_MASTERING_TRUE_PEAK_CEILING_DBTP", -1.5, -3.0, -0.5)), 3),
        "max_crest_loss_db": round(float(max_crest_loss), 3),
        "prefer_cleaner_quieter_when_target_met": bool(prefer_cleaner),
        "crest_loss_repeat_push_decay": round(float(_env_float("BUSY_BAMIX_V6314_REPEAT_CREST_LOSS_PUSH_DECAY", 0.72, 0.35, 0.95)), 3),
        "adaptive_damage_governor": {
            "schema_version": "busy_bamix_mastering_damage_governor_v8_5_3_63_2",
            "active": True,
            "mode": handoff_mode,
            "reasons": readiness_reasons,
            "policy": "v63.2 consolidates 63.1.9.x handoff and allows commercial density completion toward -9~-8 LUFS while existing v57/v60/v61/export QC remain the actual damage gates.",
        },
        "reason": "BAMix premaster is the source authority; v63.2 uses readiness plus commercial target gap to govern mastering completion instead of forced transparent polish.",
        "policy": "v63.1.10 consolidated handoff + v63.2 commercial premaster completion: transparent when already ready, density-completion when structurally safe but under-commercial, guarded/fragile when risks are present.",
    }


def _apply_bamix_mastering_handoff_policy(features: dict[str, Any], decision: dict[str, Any], bamix_report: dict[str, Any], log_fp) -> None:
    if not isinstance(features, dict) or not isinstance(decision, dict) or not isinstance(bamix_report, dict):
        return
    if not bamix_report.get("premaster_used_for_mastering"):
        return
    max_intensity = _normalize_final_mastering_intensity(os.environ.get("BUSY_AUTOMIX_MASTERING_MAX_INTENSITY", "balanced"))
    current = _normalize_final_mastering_intensity((decision.get("auto_intensity_policy") or {}).get("final_mode") if isinstance(decision.get("auto_intensity_policy"), dict) else decision.get("mastering_intensity") or os.environ.get("BUSY_MASTERING_INTENSITY") or "balanced")
    final = current
    capped = False
    if _mode_rank(current) > _mode_rank(max_intensity):
        final = max_intensity
        capped = True

    hp = _derive_bamix_v6319_commercial_handoff(bamix_report, current_mode=current, max_intensity=max_intensity, capped=capped)
    hp["final_mode_after_cap"] = final

    policy = decision.get("auto_intensity_policy") if isinstance(decision.get("auto_intensity_policy"), dict) else {}
    policy = dict(policy)
    policy["bamix_premaster_handoff_policy"] = hp
    policy["final_mode_before_bamix_cap"] = current
    policy["final_mode"] = final
    policy["bamix_premaster_cap_active"] = True
    policy["selected_true_peak_ceiling_dbtp"] = hp.get("true_peak_ceiling_dbtp")
    policy["selected_max_boost_db"] = hp.get("max_finish_push_db")
    policy["selected_chase_strength"] = hp.get("chase_strength")
    policy["bamix_effective_target_lufs"] = hp.get("effective_final_target_lufs")
    policy["bamix_effective_floor_lufs"] = hp.get("effective_floor_lufs")
    policy["bamix_handoff_mode"] = hp.get("handoff_mode")
    decision["auto_intensity_policy"] = policy
    decision["mastering_intensity"] = final
    decision.setdefault("targets", {})["bamix_mastering_target_lufs"] = hp.get("effective_final_target_lufs")
    decision.setdefault("targets", {})["bamix_mastering_floor_lufs"] = hp.get("effective_floor_lufs")
    decision["busy_auto_mixing_mastering_handoff_policy"] = hp
    decision["busy_auto_mixing_mastering_input_authority"] = True
    decision["busy_auto_mixing_premaster_is_mastering_source_authority"] = True
    features["auto_intensity_policy"] = policy
    features["busy_auto_mixing_mastering_handoff_policy"] = hp
    features["busy_auto_mixing_mastering_input_authority"] = True
    features["busy_auto_mixing_premaster_is_mastering_source_authority"] = True
    bamix_report["mastering_handoff_policy"] = hp
    bamix_report["mastering_input_authority"] = True
    bamix_report["handoff_integrity"] = {
        "schema_version": "busy_auto_mixing_handoff_integrity_v8_5_3_63_1_9",
        "premaster_used_for_mastering": True,
        "mastering_input_source": "busy_auto_mixing_premaster",
        "source_authority": True,
        "handoff_mode": hp.get("handoff_mode"),
        "commercial_readiness": hp.get("commercial_readiness"),
        "policy": "BAMix source authority is preserved while mastering is allowed to complete commercial density only when readiness permits it.",
    }
    flags = list(decision.get("processing_flags") or []) if isinstance(decision.get("processing_flags"), list) else []
    for flag in [
        "busy_auto_mixing_mastering_handoff_policy",
        "busy_auto_mixing_source_authority",
        "busy_auto_mixing_readiness_aware_handoff",
        "busy_auto_mixing_handoff_governance_not_hard_brake",
        f"busy_auto_mixing_handoff_mode_{hp.get('handoff_mode')}",
    ]:
        if flag and flag not in flags:
            flags.append(flag)
    decision["processing_flags"] = flags
    _apply_mastering_intensity_env_to_process(final, policy=policy)
    # Env handoff for audio_engine.mastering.  Values are dynamic per run and
    # mode-aware; they intentionally supersede older transparent-only defaults.
    os.environ["BUSY_AUTOMIX_PREMASTER_ACTIVE"] = "1"
    os.environ["BUSY_BAMIX_V6319_HANDOFF_MODE"] = str(hp.get("handoff_mode") or "")
    os.environ["BUSY_AUTOMIX_MASTERING_EFFECTIVE_TARGET_LUFS"] = str(hp.get("effective_final_target_lufs"))
    os.environ["BUSY_AUTOMIX_MASTERING_EFFECTIVE_FLOOR_LUFS"] = str(hp.get("effective_floor_lufs"))
    os.environ["BUSY_AUTOMIX_MASTERING_MAX_CREST_LOSS_DB"] = str(hp.get("max_crest_loss_db"))
    if hp.get("premaster_lra_lu") is not None:
        os.environ["BUSY_AUTOMIX_MASTERING_PREMASTER_LRA_LU"] = str(hp.get("premaster_lra_lu"))
    if hp.get("premaster_crest_factor_db") is not None:
        os.environ["BUSY_AUTOMIX_MASTERING_PREMASTER_CREST_DB"] = str(hp.get("premaster_crest_factor_db"))
    os.environ["BUSY_BAMIX_SELECTED_RECIPE"] = str(hp.get("selected_mix_recipe") or "")
    os.environ["BUSY_BAMIX_PUNCH_PRESERVED"] = "1" if hp.get("punch_preserved") else "0"
    os.environ["BUSY_COMMERCIAL_FINISH_MAX_PUSH_DB"] = str(hp.get("max_finish_push_db"))
    os.environ["BUSY_COMMERCIAL_DB_TARGET_MAX_PUSH_DB"] = str(hp.get("max_finish_push_db"))
    os.environ["BUSY_AUTO_INTENSITY_CHASE_STRENGTH"] = str(hp.get("chase_strength"))
    os.environ["BUSY_TRUE_PEAK_CEILING_DB"] = str(hp.get("true_peak_ceiling_dbtp"))
    _log(log_fp, "stage_done_busy_auto_mixing_mastering_handoff_policy", current=current, final=final, max_intensity=max_intensity, capped=capped, handoff_mode=hp.get("handoff_mode"), readiness=hp.get("commercial_readiness"), effective_target_lufs=hp.get("effective_final_target_lufs"), max_finish_push_db=hp.get("max_finish_push_db"), max_crest_loss_db=hp.get("max_crest_loss_db"))
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _rss_mb() -> float:
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except Exception:
        return -1.0


def _log(fp, step: str, **data: Any) -> None:
    payload = {"ts_utc": _now(), "step": step, "rss_mb": _rss_mb(), "build_id": os.environ.get("BUSY_BUILD_ID", "unknown"), **data}
    line = "[BUSY_JOB] " + json.dumps(payload, ensure_ascii=False, default=str)
    print(line, flush=True)
    if fp:
        fp.write(line + "\n")
        fp.flush()




def _env_on(name: str, default: str = "0") -> bool:
    return str(os.environ.get(name, default)).strip().lower() in {"1", "true", "yes", "on", "always"}


def _env_float(name: str, default: float, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        value = float(str(os.environ.get(name, default)).strip())
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return float(value)


def _audio_judge_key(result: dict[str, Any] | None, fallback: str) -> str:
    r = result if isinstance(result, dict) else {}
    model = str(r.get("model") or fallback).strip().lower().replace("/", "_").replace("-", "_").replace(".", "_")
    if "gemini" in model:
        return "gemini_flash_lite" if "flash" in model and "lite" in model else "gemini_audio"
    if "realtime" in model:
        return "openai_realtime_mini" if "mini" in model else "openai_realtime"
    return fallback


def _normalize_mix_for_compare(mix: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if not isinstance(mix, list):
        return out
    for item in mix:
        if not isinstance(item, dict):
            continue
        p = str(item.get("profile") or "").strip()
        if not p:
            continue
        try:
            w = float(item.get("weight") or 0.0)
        except Exception:
            w = 0.0
        out[p] = out.get(p, 0.0) + max(0.0, w)
    total = sum(out.values()) or 1.0
    return {k: round(v / total, 4) for k, v in out.items()}


def _audio_judge_council(judges: dict[str, Any]) -> dict[str, Any]:
    """Summarize dual audio element extractors.

    v8.5.0 audio judges no longer vote on genre_mix. They provide audible
    elements; the final arbiter infers genre candidates from element
    combinations + Essentia/Rule/private DB.
    """
    available = {k: v for k, v in (judges or {}).items() if isinstance(v, dict) and v.get("available")}

    def _flatten_scores(block: Any, prefix: str = "") -> dict[str, float]:
        out: dict[str, float] = {}
        if isinstance(block, dict):
            for k, v in block.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if isinstance(v, (int, float)):
                    try:
                        out[key] = float(v)
                    except Exception:
                        pass
                elif isinstance(v, dict):
                    out.update(_flatten_scores(v, key))
        return out

    element_scores_by_judge: dict[str, dict[str, float]] = {}
    role_interpretation: dict[str, Any] = {}
    for name, judge in available.items():
        element_scores_by_judge[name] = _flatten_scores(judge.get("audible_elements") or {})
        role_interpretation[name] = judge.get("role_interpretation") or {}

    all_keys: set[str] = set()
    for scores in element_scores_by_judge.values():
        all_keys.update(scores.keys())
    consensus: list[dict[str, Any]] = []
    for key in sorted(all_keys):
        vals = [scores.get(key) for scores in element_scores_by_judge.values() if key in scores]
        vals = [float(v) for v in vals if v is not None]
        if not vals:
            continue
        consensus.append({
            "element": key,
            "average_support": round(sum(vals) / len(vals), 4),
            "sources": {name: round(scores.get(key), 4) for name, scores in element_scores_by_judge.items() if key in scores},
        })
    consensus.sort(key=lambda x: float(x.get("average_support") or 0.0), reverse=True)

    # Legacy genre-mix fields are intentionally empty in element-first mode.
    return {
        "schema_version": "busy_audio_element_council_v8_5_0",
        "enabled": bool(judges),
        "available_count": len(available),
        "judge_keys": list((judges or {}).keys()),
        "available_judges": list(available.keys()),
        "judge_role": "audible_element_extractors_not_genre_classifiers",
        "element_consensus": consensus[: int(os.environ.get("BUSY_AUDIO_ELEMENT_COUNCIL_MAX_ITEMS", "24") or "24")],
        "role_interpretation_by_judge": role_interpretation,
        "average_genre_mix": [],
        "primary_profiles": {},
        "pairwise_genre_mix_overlap_first_two": None,
        "policy": "Audio judges extract audible elements only. GPT final arbiter derives genre thesis from element combinations plus Essentia/Rule/private DB.",
    }




def _stage_job_id_from_path(path: str | None) -> str:
    """Best-effort extraction of jobs/<job_id>/... for state integrity logs."""
    if not path:
        return ""
    parts = str(path).replace("\\", "/").split("/")
    try:
        i = parts.index("jobs")
        return parts[i + 1] if i + 1 < len(parts) else ""
    except Exception:
        return ""



def _post_decision_validation_council(validators: dict[str, Any]) -> dict[str, Any]:
    available = {k: v for k, v in (validators or {}).items() if isinstance(v, dict) and v.get("available")}
    verdicts = {k: str(v.get("verdict") or "uncertain") for k, v in available.items()}
    scores: list[float] = []
    for v in available.values():
        try:
            scores.append(float(v.get("support_score") or 0.0))
        except Exception:
            pass
    demote: list[dict[str, Any]] = []
    promote: list[dict[str, Any]] = []
    missed_roles: list[Any] = []
    prior_quality: dict[str, Any] = {}
    for name, v in available.items():
        prior_quality[name] = v.get("private_prior_quality")
        for item in v.get("should_demote_profiles", []) or []:
            if isinstance(item, dict):
                demote.append({"validator": name, **item})
        for item in v.get("should_promote_profiles", []) or []:
            if isinstance(item, dict):
                promote.append({"validator": name, **item})
        for item in v.get("missed_roles", []) or []:
            missed_roles.append({"validator": name, "role": item})
    avg = round(sum(scores) / len(scores), 4) if scores else 0.0
    strong_disagree = any(v in {"disagree"} for v in verdicts.values()) or (len(scores) >= 2 and avg < 0.55)
    partial_or_better = all(v in {"agree", "partial_agree"} for v in verdicts.values()) if verdicts else False
    partial_count = sum(1 for v in verdicts.values() if v == "partial_agree")
    demote_count = len(demote)
    missed_count = len(missed_roles)
    promote_count = len(promote)
    revision_reasons: list[str] = []
    if strong_disagree:
        revision_reasons.append("strong_disagreement_or_low_support")
    if demote_count >= int(os.environ.get("BUSY_POST_VALIDATION_REVISION_DEMOTE_COUNT", "1") or "1"):
        revision_reasons.append("validator_demote_request")
    if partial_count and missed_count >= int(os.environ.get("BUSY_POST_VALIDATION_REVISION_MISSED_ROLE_COUNT", "1") or "1"):
        revision_reasons.append("partial_agree_with_missed_roles")
    if partial_count and promote_count >= int(os.environ.get("BUSY_POST_VALIDATION_REVISION_PROMOTE_COUNT", "1") or "1"):
        revision_reasons.append("partial_agree_with_promote_request")
    if partial_count >= 2 and avg < _safe_float(os.environ.get("BUSY_POST_VALIDATION_PARTIAL_MIN_SUPPORT"), 0.78):
        revision_reasons.append("dual_partial_low_support")
    return {
        "schema_version": "busy_post_decision_audio_validation_council_v1_2",
        "enabled": bool(validators),
        "available_count": len(available),
        "validators": list((validators or {}).keys()),
        "available_validators": list(available.keys()),
        "verdicts": verdicts,
        "average_support_score": avg,
        "private_prior_quality": prior_quality,
        "strong_disagreement": bool(strong_disagree),
        "all_partial_or_better": bool(partial_or_better),
        "partial_agree_count": partial_count,
        "should_promote_profiles": promote[:12],
        "should_demote_profiles": demote[:12],
        "missed_roles": missed_roles[:12],
        "revision_reasons": sorted(set(revision_reasons)),
        "requires_revision": bool(revision_reasons),
        "policy": "Validation pass is strict: GPT-mini is a first text opinion, Gemini/Realtime are audio opinions, and high-arbitration plus deterministic QC should resolve material disagreement; no hot candidate may pass with unresolved problem sets.",
    }


# ---------------------------------------------------------------------------
# v8.5.0: element-first audio judge + conditional candidate validation helpers.
# ---------------------------------------------------------------------------

def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _compact_pre_audio_evidence(features: dict[str, Any], evidence: dict[str, Any] | None) -> dict[str, Any]:
    ev = evidence if isinstance(evidence, dict) else {}
    role_ev = ev.get("hybrid_role_evidence") if isinstance(ev.get("hybrid_role_evidence"), dict) else {}
    metrics = ev.get("metrics") if isinstance(ev.get("metrics"), dict) else {}
    atoms = ev.get("evidence_atoms") if isinstance(ev.get("evidence_atoms"), dict) else {}
    return {
        "schema_version": "busy_element_first_audio_judge_context_v8_5_0",
        "policy": "Objective Essentia/Rule/DSP evidence is supplied as an element checklist only. Audio judges must not output genre names or treat profile/group candidates as labels.",
        "tempo_bpm": ev.get("tempo_bpm"),
        "metrics": {k: metrics.get(k) for k in [
            "transient_density_per_sec", "four_on_floor_confidence", "side_ratio_lowband",
            "side_ratio_8k_14k", "correlation_avg", "essentia_guitar_texture_score",
            "essentia_electronic_texture_score", "essentia_noisy_industrial_score",
            "tf_voice_probability", "tf_synth_probability", "tf_electronic_probability",
            "tf_non_electronic_probability", "tf_danceable_probability",
        ] if k in metrics},
        "rule_feature_atoms": {k: atoms.get(k) for k in [
            "rhythm_grid", "density", "electronic_production", "band_instrument",
            "club_arrangement", "rock_arrangement", "industrial_texture", "acoustic_natural",
            "alt_heavy_mood", "pop_accessible_mood",
        ] if k in atoms},
        "role_evidence": role_ev.get("roles") if isinstance(role_ev.get("roles"), dict) else {},
        "genre_names_hidden_from_audio_judges": True,
        "interpretation_policy": {
            "four_on_floor": "rhythm evidence only, not a genre label",
            "electronic_texture": "production surface unless combined with other elements by final arbiter",
            "audio_judge_output": "audible elements only; final GPT infers genre thesis",
        },
    }



def _mix_overlap(a_mix: Any, b_mix: Any) -> float:
    a = _normalize_mix_for_compare(a_mix)
    b = _normalize_mix_for_compare(b_mix)
    if not a or not b:
        return 0.0
    return round(sum(min(a.get(p, 0.0), b.get(p, 0.0)) for p in set(a) | set(b)), 4)


def _post_decision_validation_triggers(features: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    council = features.get("audio_judge_council") if isinstance(features.get("audio_judge_council"), dict) else {}
    # v8.5.0: audio judges are element extractors, not genre voters.
    # Validate the final 5.4 genre thesis at least once when audio element evidence exists.
    if int(council.get("available_count") or 0) >= 1 and decision.get("genre_mix") and _env_on("BUSY_ELEMENT_THESIS_VALIDATION", "1"):
        reasons.append("element_first_candidate_validation")
    final_overlap = None

    ctx = decision.get("final_decision_context") if isinstance(decision.get("final_decision_context"), dict) else {}
    priors = ctx.get("profile_mastering_priors") if isinstance(ctx.get("profile_mastering_priors"), list) else []
    if len(priors) > int(os.environ.get("BUSY_VALIDATION_MAX_PRIVATE_PRIORS", "10") or "10"):
        reasons.append("too_many_promoted_private_profiles")
    cap = ((ctx.get("private_reference_profile_context") or {}) if isinstance(ctx.get("private_reference_profile_context"), dict) else {}).get("private_capability_expansion")
    promoted = []
    if isinstance(cap, dict):
        promoted = cap.get("promoted_candidates") or cap.get("top") or []
    if isinstance(promoted, list) and len(promoted) > int(os.environ.get("BUSY_VALIDATION_MAX_PROMOTED_CAPABILITY", "4") or "4"):
        reasons.append("too_many_promoted_capability_profiles")

    ge = decision.get("genre_evidence") if isinstance(decision.get("genre_evidence"), dict) else (features.get("pre_audio_genre_evidence") if isinstance(features.get("pre_audio_genre_evidence"), dict) else {})
    role_block = (ge.get("hybrid_role_evidence") or {}) if isinstance(ge, dict) else {}
    roles = role_block.get("roles") if isinstance(role_block, dict) and isinstance(role_block.get("roles"), dict) else {}
    role_text = json.dumps({
        "genre_mix": decision.get("genre_mix"),
        "mastering_role_directives": decision.get("mastering_role_directives"),
        "profile_blend": decision.get("profile_blend"),
        "dynamic_eq": decision.get("dynamic_eq"),
    }, ensure_ascii=False).lower()
    role_alias = {
        "riff_low_end_articulation": ["riff", "chug", "pick", "low_end", "low-end"],
        "vocal_front_focus": ["vocal", "front", "presence"],
        "space_texture_preservation": ["space", "ambient", "side", "reverb"],
        "electronic_layer_density": ["electronic", "synth", "glitch", "layer"],
        "top_texture_guard_need": ["top", "shimmer", "sibilance", "brittle"],
        "low_end_tightness_need": ["low", "kick", "bass", "sub"],
    }
    strong_missing = []
    for role, aliases in role_alias.items():
        if _safe_float(roles.get(role), 0.0) >= _safe_float(os.environ.get("BUSY_VALIDATION_STRONG_ROLE_MIN"), 0.62):
            if not any(a in role_text for a in aliases):
                strong_missing.append(role)
    if strong_missing:
        reasons.append("strong_role_not_reflected")

    risk_count = 0
    for tag in features.get("heuristic_tags", []) or []:
        if isinstance(tag, str) and any(x in tag for x in ["risk", "harsh", "shimmer", "width"]):
            risk_count += 1
    if isinstance(decision.get("risk_flags"), list):
        risk_count += len(decision.get("risk_flags"))
    art = features.get("suno_artifact_analysis") if isinstance(features.get("suno_artifact_analysis"), dict) else {}
    if _safe_float(art.get("overall_risk"), 0.0) >= _safe_float(os.environ.get("BUSY_VALIDATION_ARTIFACT_RISK_MIN"), 0.78):
        risk_count += 2
    if risk_count >= int(os.environ.get("BUSY_VALIDATION_QC_RISK_COUNT", "3") or "3"):
        reasons.append("qc_risk_high")

    return {
        "enabled": bool(reasons),
        "reasons": sorted(set(reasons)),
        "final_vs_audio_council_overlap": final_overlap,
        "audio_judge_pairwise_overlap": council.get("pairwise_genre_mix_overlap_first_two"),
        "element_consensus_count": len(council.get("element_consensus") or []) if isinstance(council.get("element_consensus"), list) else 0,
        "private_prior_count": len(priors),
        "promoted_capability_count": len(promoted) if isinstance(promoted, list) else 0,
        "strong_missing_roles": strong_missing,
        "risk_count": risk_count,
        "policy": "Element-first conditional validation: validators check the final genre thesis candidate-by-candidate against audio elements, priors, roles, and QC risk.",
    }


def _post_validation_needs_high_arbitration(council: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    verdicts = council.get("verdicts") if isinstance(council.get("verdicts"), dict) else {}
    if council.get("strong_disagreement"):
        reasons.append("validation_strong_disagreement")
    if _safe_float(council.get("average_support_score"), 1.0) < _safe_float(os.environ.get("BUSY_POST_VALIDATION_HIGH_MIN_SUPPORT"), 0.50):
        reasons.append("validation_low_support")
    vals = set(str(v) for v in verdicts.values())
    if "disagree" in vals and ("agree" in vals or "partial_agree" in vals or len(vals) > 1):
        reasons.append("validators_not_reconciled")
    if len(council.get("should_demote_profiles") or []) >= int(os.environ.get("BUSY_POST_VALIDATION_HIGH_DEMOTE_COUNT", "3") or "3"):
        reasons.append("many_demotion_requests")
    return bool(reasons), sorted(set(reasons))

def _decision_profile_digest(decision: dict[str, Any] | None) -> dict[str, Any]:
    d = decision if isinstance(decision, dict) else {}
    rv7 = d.get("reference_v7", {}) if isinstance(d.get("reference_v7", {}), dict) else {}
    gc0 = d.get("genre_chain", {}) if isinstance(d.get("genre_chain", {}), dict) else {}
    return {
        "active_profile": d.get("active_profile"),
        "selected_profile": d.get("selected_profile"),
        "detected_style": d.get("detected_style"),
        "genre_id": d.get("genre_id"),
        "raw_gpt_detected_style": d.get("raw_gpt_detected_style"),
        "active_family": d.get("active_family"),
        "selected_family": gc0.get("selected_family"),
        "reference_v7_selected_profile": rv7.get("selected_profile"),
        "reference_v7_family": rv7.get("family"),
        "selection_source": rv7.get("selection_source"),
    }


def _canonicalize_decision_for_stage(decision: dict[str, Any] | None, *, stage: str = "") -> dict[str, Any]:
    """Freeze the final profile used by downstream DSP stages.

    A-1 may contain an upstream GPT detected_style plus a later DB/evidence overlay.
    Downstream mastering must follow the final selected/reference profile only; raw
    GPT text is preserved for diagnostics but must not steer genre-specific DSP.
    """
    d = dict(decision or {})
    rv7 = d.get("reference_v7", {}) if isinstance(d.get("reference_v7", {}), dict) else {}
    gc0 = d.get("genre_chain", {}) if isinstance(d.get("genre_chain", {}), dict) else {}
    raw_detected = str(d.get("detected_style") or d.get("genre_id") or "").strip()
    selected = str(rv7.get("selected_profile") or d.get("selected_profile") or raw_detected or "universal_safe_master").strip()
    family = str(rv7.get("family") or gc0.get("selected_family") or d.get("active_family") or "").strip()
    if raw_detected and raw_detected != selected:
        d.setdefault("raw_gpt_detected_style", raw_detected)
    d["active_profile"] = selected
    d["selected_profile"] = selected
    d["detected_style"] = selected
    d["genre_id"] = selected
    if family:
        d["active_family"] = family
    gc1 = dict(gc0)
    gc1["selected_profile"] = selected
    if family:
        gc1["selected_family"] = family
    d["genre_chain"] = gc1
    integ = dict(d.get("decision_integrity", {}) if isinstance(d.get("decision_integrity", {}), dict) else {})
    integ.update({
        "canonical_profile": selected,
        "canonical_family": family,
        "raw_gpt_detected_style": d.get("raw_gpt_detected_style"),
        "reference_v7_selected_profile": rv7.get("selected_profile"),
        "reference_v7_selection_source": rv7.get("selection_source"),
        "canonicalized_at_stage": stage,
    })
    d["decision_integrity"] = integ
    return d







def _attach_existing_process_research_governance(features: dict[str, Any], decision: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Attach report-driven governance without adding a new processing stage.

    This maps the source-conditioning / Element Action Manual research into the
    existing A36/A39 decision payload: important elements are protected first,
    enhanced only after current-track evidence proves a deficit, and residue
    control cannot override original-stereo evidence.
    """
    d = dict(decision or {})
    f = features if isinstance(features, dict) else {}
    contract = d.get("role_first_dsp_contract") if isinstance(d.get("role_first_dsp_contract"), dict) else {}
    manual = contract.get("element_action_manual") if isinstance(contract.get("element_action_manual"), dict) else d.get("element_action_manual") if isinstance(d.get("element_action_manual"), dict) else {}
    sea = f.get("stem_evidence_assist") if isinstance(f.get("stem_evidence_assist"), dict) else {}
    stem_prot = sea.get("protection_priorities") if isinstance(sea.get("protection_priorities"), dict) else {}
    art = f.get("suno_artifact_analysis") if isinstance(f.get("suno_artifact_analysis"), dict) else {}
    risk_scores = art.get("risk_scores") if isinstance(art.get("risk_scores"), dict) else {}

    def sf(x: Any, default: float = 0.0) -> float:
        return _safe_float(x, default)

    preserve: list[dict[str, Any]] = []
    control: list[dict[str, Any]] = []
    reduce: list[dict[str, Any]] = []
    enhance_requires_deficit: list[dict[str, Any]] = []

    if sf(stem_prot.get("vocal_presence")) >= 0.55:
        preserve.append({"element": "vocal_presence", "priority": round(sf(stem_prot.get("vocal_presence")), 4), "rule": "preserve_before_enhance"})
        control.append({"axis": "competing_presence_masking", "rule": "control_competitors_do_not_broad_cut_vocal"})
    if sf(stem_prot.get("vocal_air")) >= 0.45:
        preserve.append({"element": "vocal_air", "priority": round(sf(stem_prot.get("vocal_air")), 4), "rule": "protect_air_from_synthetic_hash_cleanup"})
    if sf(stem_prot.get("bass_fundamental")) >= 0.45:
        preserve.append({"element": "bass_fundamental", "priority": round(sf(stem_prot.get("bass_fundamental")), 4), "rule": "low_control_must_preserve_bass_ownership"})
    if sf(stem_prot.get("drum_hat_transients")) >= 0.45:
        preserve.append({"element": "hat_cymbal_transients", "priority": round(sf(stem_prot.get("drum_hat_transients")), 4), "rule": "do_not_classify_all_8_12k_as_fizz"})
    if sf(risk_scores.get("synthetic_air"), sf(risk_scores.get("brittle_8_12k_top"))) >= 0.45:
        reduce.append({"axis": "non_musical_side_air_hash", "priority": round(max(sf(risk_scores.get("synthetic_air")), sf(risk_scores.get("brittle_8_12k_top"))), 4), "rule": "reduce_only_after_A40_A69_gates"})
    if sf(art.get("overall_risk")) >= 0.55:
        control.append({"axis": "loudness_push_exposes_artifacts", "priority": round(sf(art.get("overall_risk")), 4), "rule": "clean_or_cap_before_louder_candidate"})

    # Keep existing manual visible; this block is a deterministic governance lens,
    # not a replacement for the contract generator.
    gov = {
        "schema_version": "busy_existing_process_research_governance_v8_5_3_62_9_8",
        "active": True,
        "stage": stage,
        "mode": "existing_stage_governance_only",
        "principles": [
            "prominence_is_not_boost",
            "preserve_before_enhance",
            "control_competing_or_residue_axes_before_boosting_feature_axes",
            "stem_evidence_is_advisory_only_and_never_audio_injection",
        ],
        "preserve_priorities": preserve[:12],
        "control_priorities": control[:12],
        "reduce_priorities": reduce[:12],
        "enhance_requires_deficit": enhance_requires_deficit,
        "element_action_counts": {
            "preserve": len(manual.get("preserve_actions") or []) if isinstance(manual, dict) else 0,
            "enhance": len(manual.get("enhance_actions") or []) if isinstance(manual, dict) else 0,
            "control": len(manual.get("control_actions") or []) if isinstance(manual, dict) else 0,
            "reduce": len(manual.get("reduce_actions") or []) if isinstance(manual, dict) else 0,
            "bypass": len(manual.get("bypass_reasons") or []) if isinstance(manual, dict) else 0,
        },
        "policy": "No new processing stage. This governance payload is attached to the existing A36/A39 decision so downstream A40/A69/v58/v59 modules can prefer protection/control over boost-first behavior.",
    }
    d["existing_process_research_governance"] = gov
    d.setdefault("final_decision_context", {})["existing_process_research_governance"] = gov
    f["existing_process_research_governance"] = gov
    flags = list(d.get("processing_flags") or []) if isinstance(d.get("processing_flags"), list) else []
    if "existing_process_research_governance_v6298" not in flags:
        flags.append("existing_process_research_governance_v6298")
    d["processing_flags"] = flags
    return d


def _attach_v6299_research_governance_if_missing(features: dict[str, Any] | None, decision: dict[str, Any] | None, *, mode: str, stage: str, log_fp=None) -> dict[str, Any]:
    """Attach v62.9.9 governance to the persistent stage decision.

    v62.9.9 originally attached the research-governance bundle inside the
    audio_engine render path.  In the two-stage worker the Stage B limiter loads
    the decision from stage_state, so the report could show empty v62.9.9 blocks
    even though the code existed.  This helper attaches the same deterministic
    governance payload before stage_state/report handoff, without adding any DSP
    stage or stem-audio injection.
    """
    d = dict(decision or {})
    if not _env_on("BUSY_V6299_RESEARCH_GOVERNANCE", "1"):
        d.setdefault("v62_9_9_research_governance", {"enabled": False, "active": False, "reason": "BUSY_V6299_RESEARCH_GOVERNANCE disabled", "stage": stage})
        return d
    existing = d.get("v62_9_9_research_governance")
    if isinstance(existing, dict) and existing.get("schema_version") and existing:
        return d
    try:
        from audio_engine.mastering import _attach_v6299_research_governance as _engine_attach_v6299  # type: ignore
        before = features if isinstance(features, dict) else {}
        d, gov = _engine_attach_v6299(before, d, mode or "balanced")
        if isinstance(gov, dict):
            gov["stage_state_attached"] = True
            gov["stage_state_attached_at_stage"] = stage
            d["v62_9_9_research_governance"] = gov
            d.setdefault("final_decision_context", {})["v62_9_9_research_governance"] = gov
            if isinstance(features, dict):
                features["v62_9_9_research_governance"] = gov
        flags = list(d.get("processing_flags") or []) if isinstance(d.get("processing_flags"), list) else []
        if "v6299_stage_state_governance_attached" not in flags:
            flags.append("v6299_stage_state_governance_attached")
        d["processing_flags"] = flags
        try:
            _log(
                log_fp,
                "stage_done_v6299_stage_state_governance_attach",
                stage=stage,
                active=(gov or {}).get("active") if isinstance(gov, dict) else None,
                micro=((gov or {}).get("microdynamics_lra_shape_recovery") or {}).get("active") if isinstance(gov, dict) and isinstance((gov or {}).get("microdynamics_lra_shape_recovery"), dict) else None,
                section=((gov or {}).get("section_aware_spectral_consistency") or {}).get("active") if isinstance(gov, dict) and isinstance((gov or {}).get("section_aware_spectral_consistency"), dict) else None,
                texture=((gov or {}).get("musical_texture_preservation_gate") or {}).get("active") if isinstance(gov, dict) and isinstance((gov or {}).get("musical_texture_preservation_gate"), dict) else None,
                low_end=((gov or {}).get("translation_aware_low_end_governor") or {}).get("active") if isinstance(gov, dict) and isinstance((gov or {}).get("translation_aware_low_end_governor"), dict) else None,
            )
        except Exception:
            pass
        return d
    except Exception as exc:
        fallback = {
            "schema_version": "busy_v6299_research_governance_stage_state_attach_error_v8_5_3_62_9_9_2",
            "enabled": True,
            "active": False,
            "stage": stage,
            "reason": "exception_attaching_engine_governance",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "policy": "Report-only fallback; mastering continues with existing decision.",
        }
        d["v62_9_9_research_governance"] = fallback
        d.setdefault("final_decision_context", {})["v62_9_9_research_governance"] = fallback
        if isinstance(features, dict):
            features["v62_9_9_research_governance"] = fallback
        try:
            _log(log_fp, "stage_error_v6299_stage_state_governance_attach", stage=stage, error=str(exc)[:500])
        except Exception:
            pass
        return d


def _attach_v6360_research_capability_framework_if_missing(
    features: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    *,
    stage: str,
    report: dict[str, Any] | None = None,
    log_fp=None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Attach v63.6.2 research capability matrix/router telemetry.

    The framework still prevents per-label hotfixes and warning suppression.
    v63.6.1.1 may expose a guarded drum_transient_punch DSP result when the
    audio engine applies that capability; v63.6.2 additionally surfaces the
    DML/ownership/final-limiter workload contract without changing thresholds
    or chasing target LUFS.
    """
    f = features if isinstance(features, dict) else {}
    d = dict(decision or {})
    try:
        f, d, fw = apply_research_capability_framework(f, d, stage=stage, report=report)
        if isinstance(report, dict):
            report.update(attach_research_capability_to_report(report, features=f, decision=d, stage=stage))
        try:
            matrix = fw.get("remaining_capability_matrix") if isinstance(fw, dict) else []
            router = fw.get("capability_router") if isinstance(fw, dict) else {}
            routes = router.get("route_entries") if isinstance(router, dict) else []
            _log(
                log_fp,
                "stage_done_v6362_research_capability_framework",
                stage=stage,
                active=bool(fw.get("active")) if isinstance(fw, dict) else False,
                matrix_count=len(matrix or []) if isinstance(matrix, list) else 0,
                route_count=len(routes or []) if isinstance(routes, list) else 0,
                top_routes=[r.get("capability_id") for r in (routes or [])[:5] if isinstance(r, dict)],
            )
        except Exception:
            pass
        return f, d, fw
    except Exception as exc:
        fallback = {
            "schema_version": RESEARCH_CAPABILITY_SCHEMA_VERSION,
            "engine_version": RESEARCH_CAPABILITY_ENGINE_VERSION,
            "enabled": True,
            "active": False,
            "stage": stage,
            "reason": "exception_attaching_research_capability_framework",
            "error_type": type(exc).__name__,
            "error": str(exc)[:500],
            "policy": "Report-only fallback; mastering continues with existing audio-render behavior; v63.6.2 framework fallback does not suppress warnings, change thresholds, or chase target LUFS.",
        }
        f["research_capability_framework"] = fallback
        d["research_capability_framework"] = fallback
        d.setdefault("final_decision_context", {})["research_capability_framework"] = fallback
        try:
            _log(log_fp, "stage_error_v6362_research_capability_framework", stage=stage, error=str(exc)[:500])
        except Exception:
            pass
        return f, d, fallback

def _analysis_sources_snapshot(features: dict[str, Any] | None, decision: dict[str, Any] | None) -> dict[str, Any]:
    """Full source-by-source genre/analysis record kept in stage_state/report.

    The decision object is the fusion result.  This block preserves the separate
    inputs so we can debug whether Realtime, Essentia, Genre Evidence, or user
    hint caused a wrong profile selection.
    """
    f = features if isinstance(features, dict) else {}
    d = decision if isinstance(decision, dict) else {}
    ess = f.get("essentia_evidence") if isinstance(f.get("essentia_evidence"), dict) else {}
    ess_scores = ess.get("evidence_scores") if isinstance(ess.get("evidence_scores"), dict) else {}
    tf = ess.get("tensorflow_tags") if isinstance(ess.get("tensorflow_tags"), dict) else {}
    tf_summary = tf.get("summary_scores") if isinstance(tf.get("summary_scores"), dict) else {}
    tf_mood_theme = tf.get("mood_theme") if isinstance(tf.get("mood_theme"), dict) else {}
    tf_danceability = tf.get("danceability") if isinstance(tf.get("danceability"), dict) else {}
    rt = f.get("realtime_audio_judge") if isinstance(f.get("realtime_audio_judge"), dict) else {}
    preview = f.get("realtime_genre_preview") if isinstance(f.get("realtime_genre_preview"), dict) else {}
    ge = d.get("genre_evidence") if isinstance(d.get("genre_evidence"), dict) else {}
    hint = f.get("user_style_hint") if isinstance(f.get("user_style_hint"), dict) else {}
    ref = d.get("reference_v7") if isinstance(d.get("reference_v7"), dict) else {}
    gc0 = d.get("genre_chain") if isinstance(d.get("genre_chain"), dict) else {}
    return {
        "schema_version": "busy_analysis_sources_v1",
        "generated_at_utc": _now(),
        "source_policy": "Realtime/Essentia/GenreEvidence/UserHint are stored separately; final decision is the fusion result, not a replacement for these records.",
        "final_fusion_decision": {
            "selected_profile": d.get("selected_profile") or d.get("active_profile") or ref.get("selected_profile"),
            "selected_family": d.get("active_family") or gc0.get("selected_family") or ref.get("family"),
            "detected_style": d.get("detected_style") or d.get("genre_id"),
            "classification_mode": d.get("classification_mode"),
            "genre_mix": d.get("genre_mix"),
            "profile_blend": d.get("profile_blend"),
            "hybrid_profile_label": d.get("hybrid_profile_label"),
            "blend_strategy": d.get("blend_strategy"),
            "raw_gpt_detected_style": d.get("raw_gpt_detected_style"),
            "confidence": d.get("style_confidence") or d.get("confidence"),
            "reference_v7_selected_profile": ref.get("selected_profile"),
            "reference_v7_family": ref.get("family"),
            "selection_source": ref.get("selection_source"),
            "model_used": d.get("model_used"),
            "model_router": d.get("model_router"),
            "high_accuracy_escalated": d.get("high_accuracy_escalated"),
            "final_arbitration": d.get("final_arbitration", {}),
            "final_decision_context": d.get("final_decision_context", {}),
            "detailed_genre_decision": d.get("detailed_genre_decision", {}),
        },
        "user_style_hint": {
            "provided": bool(hint.get("provided")),
            "raw_text": hint.get("raw_text", ""),
            "trust_policy": hint.get("trust_policy", "hint_may_be_absent_or_wrong_cross_checked_against_audio"),
            "group_scores": ge.get("user_hint_group_scores", {}),
            "hits": ge.get("user_hint_hits", {}),
            "normalized": d.get("detailed_genre_decision", {}).get("user_style_hint_normalization", {}) if isinstance(d.get("detailed_genre_decision"), dict) else {},
            "validated_user_hints": d.get("detailed_genre_decision", {}).get("validated_user_hints", []) if isinstance(d.get("detailed_genre_decision"), dict) else [],
            "rejected_user_hints": d.get("detailed_genre_decision", {}).get("rejected_user_hints", []) if isinstance(d.get("detailed_genre_decision"), dict) else [],
        },
        "genre_evidence_engine_full": {
            "available": bool(ge),
            "schema_version": ge.get("schema_version"),
            "tempo_bpm": ge.get("tempo_bpm"),
            "top_profile": ge.get("top_profile"),
            "top_score": ge.get("top_score"),
            "audio_group_scores": ge.get("audio_group_scores", {}),
            "fused_group_scores": ge.get("fused_group_scores", {}),
            "profile_scores_v6": ge.get("profile_scores_v6", {}),
            "profile_candidates": ge.get("profile_candidates", []),
            "rejected_candidates": ge.get("rejected_candidates", []),
            "metrics": ge.get("metrics", {}),
            "private_reference_db": ge.get("private_reference_db", {}),
            "db_profile_scores": ge.get("db_profile_scores", {}),
            "db_profile_candidates": ge.get("db_profile_candidates", []),
            "top_db_profile": ge.get("top_db_profile"),
            "top_db_score": ge.get("top_db_score"),
        },
        "essentia": {
            "enabled": ess.get("enabled"),
            "available": bool(ess.get("available")),
            "reason": ess.get("reason"),
            "package": ess.get("package"),
            "seconds_analyzed": ess.get("seconds_analyzed") or ess.get("seconds"),
            "sample_rate": ess.get("sample_rate"),
            "frame_count": ess.get("frame_count"),
            "spectral_centroid_mean_hz": ess.get("spectral_centroid_mean_hz"),
            "spectral_rolloff_mean_hz": ess.get("spectral_rolloff_mean_hz"),
            "spectral_flatness_mean": ess.get("spectral_flatness_mean"),
            "spectral_flux_mean": ess.get("spectral_flux_mean"),
            "zero_crossing_rate_mean": ess.get("zero_crossing_rate_mean"),
            "key": ess.get("key", {}),
            "evidence_scores": ess_scores,
            "error_type": ess.get("error_type"),
            "error": ess.get("error"),
        },
        "essentia_tensorflow": {
            "enabled": tf.get("enabled"),
            "available": bool(tf.get("available")),
            "reason": tf.get("reason"),
            "schema_version": tf.get("schema_version"),
            "model_status": tf.get("model_status", {}),
            "summary_scores": tf_summary,
            "instrument": tf.get("instrument", {}),
            "electronic": tf.get("electronic", {}),
            "voice_instrumental": tf.get("voice_instrumental", {}),
            "mood_theme": tf_mood_theme,
            "danceability": tf_danceability,
            "mood_theme_summary": {
                "happy": tf_summary.get("mood_happy_probability"),
                "upbeat": tf_summary.get("mood_upbeat_probability"),
                "commercial": tf_summary.get("mood_commercial_probability"),
                "party": tf_summary.get("mood_party_probability"),
                "emotional": tf_summary.get("mood_emotional_probability"),
                "dark": tf_summary.get("mood_dark_probability"),
                "aggressive": tf_summary.get("mood_aggressive_probability"),
                "relaxed": tf_summary.get("mood_relaxed_probability"),
                "danceable": tf_summary.get("danceable_probability"),
                "pop_mood_score": tf_summary.get("pop_mood_score"),
                "alt_rock_mood_score": tf_summary.get("alt_rock_mood_score"),
                "ballad_mood_score": tf_summary.get("ballad_mood_score"),
            },
            "variable_evidence_mix": tf.get("variable_evidence_mix"),
            "error_type": tf.get("error_type"),
            "error": tf.get("error"),
        },
        "realtime_audio_judge": {
            "enabled": str(os.environ.get("BUSY_REALTIME_GENRE_JUDGE", "1")),
            "available": bool(rt.get("available")),
            "reason": rt.get("reason"),
            "model": rt.get("model"),
            "classification_mode": rt.get("classification_mode"),
            "selected_profile": rt.get("selected_profile"),
            "selected_family": rt.get("selected_family"),
            "confidence": rt.get("confidence"),
            "genre_mix": rt.get("genre_mix", []),
            "genre_mix_policy": rt.get("genre_mix_policy", {}),
            "mastering_role_observations": rt.get("mastering_role_observations", []),
            "audible_family": rt.get("audible_family"),
            "audible_evidence": rt.get("audible_evidence", []),
            "rejected_candidates": rt.get("rejected_candidates", []),
            "candidate_profiles": rt.get("candidate_profiles", []),
            "raw_selected_profile": rt.get("raw_selected_profile"),
            "raw_genre_mix": rt.get("raw_genre_mix"),
            "raw_text": rt.get("raw_text"),
            "raw_response": rt.get("raw_response"),
            "raw_events_seen": rt.get("events_seen", []),
            "parse_error": rt.get("parse_error"),
            "notes": rt.get("notes"),
            "api_usage": rt.get("api_usage", {}),
            "error": rt.get("error"),
            "preview": rt.get("preview") or preview,
        },
        "gemini_audio_judge": {
            "enabled": str(os.environ.get("BUSY_GEMINI_AUDIO_JUDGE", "1")),
            **({} if not isinstance(f.get("gemini_audio_judge"), dict) else f.get("gemini_audio_judge")),
        },
        "audio_judges": f.get("audio_judges", {}),
        "audio_judge_council": f.get("audio_judge_council", {}),
        "evidence_informed_audio_judge_context": f.get("evidence_informed_audio_judge_context", {}),
        "pre_audio_genre_evidence": f.get("pre_audio_genre_evidence", {}),
        "post_decision_audio_validation": f.get("post_decision_audio_validation", {}),
        "genre_evidence_engine": {
            "top_profile": ge.get("top_profile"),
            "top_score": ge.get("top_score"),
            "profile_candidates": ge.get("profile_candidates", []),
            "db_profile_candidates": ge.get("db_profile_candidates", []),
            "private_reference_db": ge.get("private_reference_db", {}),
            "db_profile_scores": ge.get("db_profile_scores", {}),
            "top_db_profile": ge.get("top_db_profile"),
            "top_db_score": ge.get("top_db_score"),
            "variable_evidence_mix": ge.get("variable_evidence_mix"),
            "hybrid_role_evidence": ge.get("hybrid_role_evidence"),
            "audio_group_scores": ge.get("audio_group_scores", {}),
            "fused_group_scores": ge.get("fused_group_scores", {}),
        },
    }

def _stage_state_snapshot_for_report(state: dict[str, Any] | None, decision: dict[str, Any] | None) -> dict[str, Any]:
    """Compact state snapshot embedded into final reports/DB rows.

    Supabase stage_state.json is a temporary processing artifact and may be
    cleaned after success. This snapshot makes the final report self-contained
    so learning logs always know which A-1 analysis/profile actually drove DSP.
    """
    st = state if isinstance(state, dict) else {}
    dec = decision if isinstance(decision, dict) else {}
    return {
        "state_schema": st.get("state_schema"),
        "stage": st.get("stage"),
        "job_id": st.get("job_id"),
        "source_input_supabase_path": st.get("source_input_supabase_path"),
        "preprocessed_supabase_path": st.get("preprocessed_supabase_path"),
        "a2a_supabase_path": st.get("a2a_supabase_path"),
        "intermediate_supabase_path": st.get("intermediate_supabase_path"),
        "stage_a1_completed_at_utc": st.get("stage_a1_completed_at_utc"),
        "stage_a2a_completed_at_utc": st.get("stage_a2a_completed_at_utc"),
        "stage_a2b_completed_at_utc": st.get("stage_a2b_completed_at_utc"),
        "decision_digest": st.get("decision_digest") or _decision_profile_digest(dec),
        "decision_integrity": dec.get("decision_integrity", {}),
        "input_analysis_keys": sorted(list((st.get("input_analysis") or {}).keys())) if isinstance(st.get("input_analysis"), dict) else [],
        "has_essentia_evidence": bool(((st.get("input_analysis") or {}).get("essentia_evidence") if isinstance(st.get("input_analysis"), dict) else None)),
        "has_essentia_tensorflow": bool((((st.get("input_analysis") or {}).get("essentia_evidence") or {}).get("tensorflow_tags") if isinstance((st.get("input_analysis") or {}).get("essentia_evidence"), dict) else None)) if isinstance(st.get("input_analysis"), dict) else False,
        "has_realtime_audio_judge": bool(((st.get("input_analysis") or {}).get("realtime_audio_judge") if isinstance(st.get("input_analysis"), dict) else None)),
        "has_gemini_audio_judge": bool(((st.get("input_analysis") or {}).get("gemini_audio_judge") if isinstance(st.get("input_analysis"), dict) else None)),
        "has_audio_judges": bool(((st.get("input_analysis") or {}).get("audio_judges") if isinstance(st.get("input_analysis"), dict) else None)),
        "has_genre_evidence_engine": bool(((st.get("decision") or {}).get("genre_evidence") if isinstance(st.get("decision"), dict) else None)),
        "has_analysis_sources": bool(st.get("analysis_sources")),
        "has_spatial_fx_evidence": bool(((st.get("input_analysis") or {}).get("spatial_fx_evidence") if isinstance(st.get("input_analysis"), dict) else None)),
        "has_research_capability_framework": bool(st.get("research_capability_framework") or dec.get("research_capability_framework") or (((st.get("input_analysis") or {}).get("research_capability_framework")) if isinstance(st.get("input_analysis"), dict) else None)),
        "research_capability_schema": ((st.get("research_capability_framework") or {}).get("schema_version") if isinstance(st.get("research_capability_framework"), dict) else None) or ((dec.get("research_capability_framework") or {}).get("schema_version") if isinstance(dec.get("research_capability_framework"), dict) else None),
        "remaining_capability_count": len(st.get("remaining_capability_matrix") or []) if isinstance(st.get("remaining_capability_matrix"), list) else 0,
        "capability_router_active": bool(((st.get("capability_router") or {}).get("active") if isinstance(st.get("capability_router"), dict) else False) or (((dec.get("capability_router") or {}).get("active")) if isinstance(dec.get("capability_router"), dict) else False)),
    }


def _force_report_to_stage_decision(report: dict[str, Any], state: dict[str, Any] | None, decision: dict[str, Any], *, stage: str) -> dict[str, Any]:
    """Ensure final report and DB learning use the same profile as DSP.

    This prevents an upstream GPT detected_style or stale diagnostic text from
    being recorded as if it were the active mastering profile.
    """
    out = dict(report or {})
    d = _canonicalize_decision_for_stage(decision, stage=stage)
    digest = _decision_profile_digest(d)
    summary = out.get("decision_summary") if isinstance(out.get("decision_summary"), dict) else {}
    summary = dict(summary)
    summary.update({
        "detected_style": d.get("detected_style") or d.get("selected_profile"),
        "selected_profile": d.get("selected_profile"),
        "style_confidence": d.get("style_confidence"),
        "boldness_level": d.get("boldness_level", "auto"),
        "profile_blend": d.get("profile_blend"),
        "mastering_intent": d.get("mastering_intent"),
        "reference_v7": d.get("reference_v7"),
        "raw_gpt_detected_style": d.get("raw_gpt_detected_style"),
        "decision_integrity": d.get("decision_integrity", {}),
        "detailed_genre_decision": d.get("detailed_genre_decision", {}),
    })
    out["decision_summary"] = summary
    out["genre_chain"] = d.get("genre_chain", out.get("genre_chain", {}))
    out["reference_v7"] = d.get("reference_v7", out.get("reference_v7", {}))
    out["stage_state_snapshot"] = _stage_state_snapshot_for_report(state, d)
    if isinstance(state, dict) and isinstance(state.get("analysis_sources"), dict):
        out["analysis_sources"] = state.get("analysis_sources")
    else:
        out["analysis_sources"] = _analysis_sources_snapshot((state or {}).get("input_analysis") if isinstance(state, dict) else {}, d)
    out["stage_state_decision_digest"] = digest
    out["analysis_state_source"] = "supabase_stage_state_json"
    return out



def _v6360_2_stage_state_lineage_fields(
    report: dict[str, Any] | None,
    state: dict[str, Any] | None = None,
    *,
    stage: str,
) -> dict[str, Any]:
    """Return top-level lineage fields for public stage_state.

    v63.6.0.2 fixed null base_* lineage mirroring.  v63.6.1.1 keeps that
    separation while allowing the base audio-render engine to advance to the
    guarded drum_transient_punch renderer and the wrapper engine to remain the
    research capability router.  Supabase/debug review can distinguish:

    - base audio-render engine (v63.6.1 drum/transient/punch when present)
    - v63.6.2 research framework/report wrapper
    - whether lineage was resolved from the final report or is still pending
    """
    rep = report if isinstance(report, dict) else {}
    st = state if isinstance(state, dict) else {}

    def _clean(v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, str):
            v = v.strip()
            return v or None
        return v

    current_engine = _clean(rep.get("engine_version")) or _clean(st.get("engine_version"))
    current_schema = _clean(rep.get("mastering_report_schema_version")) or _clean(st.get("mastering_report_schema_version"))
    base_engine = _clean(rep.get("base_engine_version")) or _clean(st.get("base_engine_version"))
    base_schema = _clean(rep.get("base_mastering_report_schema_version")) or _clean(st.get("base_mastering_report_schema_version"))

    if not base_engine and current_engine and current_engine != RESEARCH_CAPABILITY_ENGINE_VERSION:
        base_engine = current_engine
    if not base_schema and current_schema and current_schema != RESEARCH_CAPABILITY_REPORT_SCHEMA:
        base_schema = current_schema

    resolved = bool(base_engine or base_schema)
    fields: dict[str, Any] = {
        "engine_version": RESEARCH_CAPABILITY_ENGINE_VERSION,
        "mastering_report_schema_version": RESEARCH_CAPABILITY_REPORT_SCHEMA,
        "framework_engine_version": RESEARCH_CAPABILITY_ENGINE_VERSION,
        "framework_mastering_report_schema_version": RESEARCH_CAPABILITY_REPORT_SCHEMA,
        "research_capability_lineage": {
            "schema_version": "busy_research_capability_lineage_v8_5_3_63_6_2_11",
            "active": True,
            "stage": stage,
            "lineage_resolved": resolved,
            "lineage_source": "final_report" if (rep and resolved) else "pending_final_report",
            "base_engine_version": base_engine,
            "base_mastering_report_schema_version": base_schema,
            "framework_engine_version": RESEARCH_CAPABILITY_ENGINE_VERSION,
            "framework_mastering_report_schema_version": RESEARCH_CAPABILITY_REPORT_SCHEMA,
            "research_capability_schema_version": RESEARCH_CAPABILITY_SCHEMA_VERSION,
            "policy": "stage_state mirrors final report lineage; v63.6.2 framework wrapper is separated from the guarded drum_transient_punch audio-render base engine and exposes DML ownership contract telemetry.",
        },
    }
    if base_engine:
        fields["base_engine_version"] = base_engine
    elif "base_engine_version" in st:
        fields["base_engine_version"] = None
    if base_schema:
        fields["base_mastering_report_schema_version"] = base_schema
    elif "base_mastering_report_schema_version" in st:
        fields["base_mastering_report_schema_version"] = None
    return fields


def _v6360_2_sync_stage_state_lineage(
    state: dict[str, Any] | None,
    report: dict[str, Any] | None,
    *,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(state, dict):
        return {}
    fields = _v6360_2_stage_state_lineage_fields(report, state, stage=stage)
    state.update(fields)
    return fields


def _stage_b_final_report_summary(report: dict[str, Any]) -> dict[str, Any]:
    """Compact final report summary copied into Stage B stage_state.

    v8.5.3.14 keeps post-master QC, translation, confidence, AI-residue
    architecture mapper and adaptive commercial residue controls visible in the final downloadable stage_state instead
    of leaving them only in the separate report JSON.
    """
    report = report if isinstance(report, dict) else {}
    keys = [
        "output_analysis",
        "decision_summary",
        "reference_v7",
        "commercial_loudness_result",
        "commercial_loudness_finish",
        "virtual_hot_commercial_solver",
        "representative_section_map",
        "virtual_limiter_sweep",
        "blocker_capacity_map",
        "blocker_solution_plan",
        "final_push_strategy",
        "musical_role_identity_map",
        "virtual_premaster_strategy",
        "residue_pre_clean_stage",
        "original_input_analysis_before_pre_clean",
        "adaptive_residue_light_control",
        "adaptive_hot_commercial",
        "experimental_residue_light_control",
        "experimental_hot_commercial",
        "stereo_qc",
        "ab_delta_qc",
        "lra_guard",
        "v6361_drum_transient_punch",
        "drum_transient_punch",
        "v6404_commercial_maximum_density",
        "commercial_maximum_density",
        "codec_preview_guard",
        "playback_translation",
        "playback_translation_auditor",
        "post_master_qc_explainer",
        "rollback_ledger",
        "final_master_confidence",
        "adaptive_residue_light_control",
        "adaptive_hot_commercial",
        "experimental_residue_light_control",
        "experimental_hot_commercial",
        "ai_residue_architecture_mapper",
        "ai_sound_architecture_analysis",
        "residue_component_separation",
        "musicality_protection_map",
        "shadow_residue_attenuation",
        "direct_residue_action_gate",
        "ai_residue_final_user_summary",
        "remaining_safety_warnings",
        "risk_metrics",
        "dml_report_authority_stage_state_v6362_11",
        "target_semantics_v6362_10",
        "final_source_authority_v6362_10",
        "v6362_10_report_authority_finalizer",
        "v6362_dml_ownership_contract",
        "v60_deterministic_multistage_limiter",
    ]
    return {k: report.get(k, {}) for k in keys if k in report}


def _stage_b_top_level_qc_fields(report: dict[str, Any]) -> dict[str, Any]:
    """Fields worth surfacing at top level of final stage_state."""
    report = report if isinstance(report, dict) else {}
    keys = [
        "post_master_qc_explainer",
        "rollback_ledger",
        "v6361_drum_transient_punch",
        "drum_transient_punch",
        "playback_translation_auditor",
        "final_master_confidence",
        "virtual_hot_commercial_solver",
        "representative_section_map",
        "blocker_capacity_map",
        "blocker_solution_plan",
        "final_push_strategy",
        "musical_role_identity_map",
        "virtual_premaster_strategy",
        "residue_pre_clean_stage",
        "ai_residue_architecture_mapper",
        "ai_sound_architecture_analysis",
        "residue_component_separation",
        "musicality_protection_map",
        "shadow_residue_attenuation",
        "direct_residue_action_gate",
        "ai_residue_final_user_summary",
    ]
    return {k: report.get(k, {}) for k in keys if isinstance(report.get(k), (dict, list))}





def _v6362_11_stage_state_authority_mirror_fields(report: dict[str, Any] | None) -> dict[str, Any]:
    """Expose final DML/report-authority telemetry in Stage B stage_state.

    v63.6.2.10 fixed the final report/debug authority order, but Stage B
    stage_state only copied generic QC blocks and therefore could still miss the
    DML ownership/report-authority contract that the final report already had.
    This is report/state telemetry only; it does not alter audio, DML, limiter,
    target thresholds, or warning logic.
    """
    rep = report if isinstance(report, dict) else {}

    def _first_dict(*keys: str) -> dict[str, Any]:
        for key in keys:
            val = rep.get(key)
            if isinstance(val, dict):
                return dict(val)
        return {}

    target = _first_dict(
        "target_semantics_v6362_10",
        "target_semantics_v6362_9",
        "target_semantics_v6362_8",
        "target_semantics_v6362_7",
        "target_semantics_v6362_6",
        "target_semantics_v6362_5",
        "target_semantics_v6362_4",
        "target_semantics_v6362_2",
    )
    final_source_authority = _first_dict(
        "final_source_authority_v6362_10",
        "final_source_authority_v6362_9",
        "final_source_authority_v6362_8",
        "final_source_authority_v6362_7",
        "final_source_authority_v6362_6",
        "final_source_authority_v6362_5",
        "final_source_authority_v6362_4",
        "final_source_authority_v6362_3",
        "final_source_authority_v6362_2",
    )
    finalizer = _first_dict(
        "v6362_10_report_authority_finalizer",
        "v6362_9_report_authority_finalizer",
        "v6362_8_report_authority_finalizer",
    )
    v60 = rep.get("v60_deterministic_multistage_limiter") if isinstance(rep.get("v60_deterministic_multistage_limiter"), dict) else {}
    contract = rep.get("v6362_dml_ownership_contract") if isinstance(rep.get("v6362_dml_ownership_contract"), dict) else {}
    if not contract and isinstance(v60, dict):
        maybe = v60.get("v6362_dml_ownership_contract")
        if isinstance(maybe, dict):
            contract = maybe
        else:
            gate = v60.get("acceptance_gate") if isinstance(v60.get("acceptance_gate"), dict) else {}
            maybe = gate.get("v6362_dml_ownership_contract") if isinstance(gate, dict) else None
            if isinstance(maybe, dict):
                contract = maybe

    active = bool(target or final_source_authority or finalizer or contract or v60)
    mirror = {
        "schema_version": "busy_v6362_11_stage_state_authority_mirror",
        "active": active,
        "source": "final_report_authority_after_rcf_attachment",
        "report_engine_version": rep.get("engine_version"),
        "mastering_report_schema_version": rep.get("mastering_report_schema_version"),
        "final_source": rep.get("final_source"),
        "final_source_authority_reason": final_source_authority.get("resolution_reason") if final_source_authority else None,
        "target_reached": target.get("target_reached") if target else rep.get("target_reached"),
        "target_lufs_gap_lu": target.get("target_lufs_gap_lu") if target else rep.get("target_lufs_gap_lu"),
        "commercial_floor_met": target.get("commercial_floor_met") if target else rep.get("commercial_floor_met"),
        "target_miss_allowed": target.get("target_miss_allowed_by_dml_contract") if target else rep.get("target_miss_allowed"),
        # v63.7.0.1: explicit DML-named alias for smoke checks and human
        # debugging.  This mirrors the same authority as target_miss_allowed.
        "target_miss_allowed_by_dml": target.get("target_miss_allowed_by_dml_contract") if target else rep.get("target_miss_allowed_by_dml", rep.get("target_miss_allowed")),
        "dml_contract_active": bool(contract.get("active")) if isinstance(contract, dict) else False,
        "dml_contract_passed": contract.get("contract_passed") if isinstance(contract, dict) else None,
        "dml_contract_reason": contract.get("reason") if isinstance(contract, dict) else None,
        "v60_active": bool(v60.get("active")) if isinstance(v60, dict) else False,
        "v60_accepted": bool(v60.get("accepted")) if isinstance(v60, dict) else False,
        "v60_applied": bool(v60.get("applied")) if isinstance(v60, dict) else False,
        "report_authority_finalizer_active": bool(finalizer.get("active")) if finalizer else False,
        "policy": "Stage-state mirror only. Final report remains the authority; v63.7.0.1 restores explicit mirrored_keys and target_miss_allowed_by_dml aliases. No DSP, limiter gain, target threshold, warning threshold, or Drum/Transient/Punch intensity is changed.",
    }
    out: dict[str, Any] = {"dml_report_authority_stage_state_v6362_11": mirror}
    if target:
        out["target_semantics_v6362_10"] = target
    if final_source_authority:
        out["final_source_authority_v6362_10"] = final_source_authority
    if finalizer:
        out["v6362_10_report_authority_finalizer"] = finalizer
    if contract:
        out["v6362_dml_ownership_contract"] = dict(contract)
    if isinstance(v60, dict) and v60:
        out["v60_deterministic_multistage_limiter"] = dict(v60)
    mirror["mirrored_keys"] = [
        key for key in (
            "target_semantics_v6362_10",
            "final_source_authority_v6362_10",
            "v6362_10_report_authority_finalizer",
            "v6362_dml_ownership_contract",
            "v60_deterministic_multistage_limiter",
        )
        if key in out
    ]
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _clean_user_style_hint(raw: str | None) -> str:
    text = str(raw or "").replace("\x00", " ").strip()
    text = " ".join(text.split())
    return text[:1000]



def _redact_private_evidence_blocks(obj: Any) -> Any:
    """Redact private DB/capability evidence while preserving public runtime DSP fields."""
    private_keys = {
        "private_reference_db", "private_capability_expansion", "private_profile_mix_candidates",
        "promoted_capability_candidates", "promoted_capability_top", "pre_capability_top",
        "audit_candidates", "db_profile_candidates", "top_db_profile", "profile_mastering_priors",
        "private_db_profiles_used", "private_reference_profile_context", "private_profile_db",
        "private_rulepack", "private_rules", "private_bundle", "private_rulepack_runtime", "private_role_prior_snippets", "db_profile_scores",
    }
    if isinstance(obj, list):
        return [_redact_private_evidence_blocks(x) for x in obj]
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            lk = str(k)
            if lk in private_keys:
                if isinstance(v, list):
                    out[lk + "_count"] = len(v)
                elif isinstance(v, dict):
                    out[lk + "_key_count"] = len(v)
                out[lk] = {"redacted": True, "reason": "private_evidence_hidden_in_public_stage_state"}
            elif lk in {"source", "selection_source"} and isinstance(v, str) and "private" in v.lower():
                out[lk] = "private_source_redacted"
            else:
                out[k] = _redact_private_evidence_blocks(v)
        return out
    return obj


def _redact_named_profile_refs(obj: Any) -> Any:
    """Redact profile names inside validation/prior payloads for public stage_state."""
    if isinstance(obj, list):
        return [_redact_named_profile_refs(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            lk = str(k).lower()
            if lk in {"profile", "selected_profile", "primary_profile", "detected_style", "source_score"}:
                if lk == "source_score":
                    out[k] = "redacted"
                else:
                    out[k] = "redacted_profile"
            elif lk in {"private_role_prior_snippets", "db_profile_scores", "profile_mastering_priors", "private_db_profiles_used"}:
                out[str(k) + "_count"] = len(v) if isinstance(v, list) else 0
                out[k] = []
            else:
                out[k] = _redact_named_profile_refs(v)
        return out
    return obj


def _sanitize_private_decision_payload(decision: Any) -> Any:
    """Remove private profile/rulepack internals from public stage_state.

    The executable DSP fields are kept; only private reference profile names,
    raw priors, role-source internals, prompt-like context, and private snippets
    are summarized.  A-2/B continue to run from the executable chain fields.
    """
    if not isinstance(decision, dict):
        return decision
    d = deepcopy(decision)
    ctx = d.get("final_decision_context") if isinstance(d.get("final_decision_context"), dict) else None
    if isinstance(ctx, dict):
        profiles_used = ctx.get("private_db_profiles_used") if isinstance(ctx.get("private_db_profiles_used"), list) else []
        priors = ctx.get("profile_mastering_priors") if isinstance(ctx.get("profile_mastering_priors"), list) else []
        ctx["private_db_profiles_used_count"] = len(profiles_used)
        ctx["profile_mastering_priors_count"] = len(priors)
        ctx["private_prior_logging_policy"] = "redacted_in_public_stage_state_role_prior_summary_only"
        ctx.pop("private_db_profiles_used", None)
        ctx.pop("profile_mastering_priors", None)
        for key in ["private_reference_profile_context", "profile_mastering_prior", "private_reference_db", "private_rulepack", "private_rules"]:
            if key in ctx:
                ctx[key] = {"redacted": True, "reason": "private_reference_context_hidden_in_stage_state"}

        rfctx = ctx.get("role_first_dsp_contract") if isinstance(ctx.get("role_first_dsp_contract"), dict) else None
        if isinstance(rfctx, dict):
            snippets = rfctx.get("private_role_prior_snippets") if isinstance(rfctx.get("private_role_prior_snippets"), list) else []
            rfctx["private_role_prior_count"] = int(rfctx.get("private_role_prior_count") or len(snippets))
            rfctx.pop("private_role_prior_snippets", None)
            rfctx.pop("db_profile_scores", None)
            rfctx.pop("private_rulepack_runtime", None)
        pv = ctx.get("post_decision_audio_validation") if isinstance(ctx.get("post_decision_audio_validation"), dict) else None
        if isinstance(pv, dict) and _env_on("BUSY_REDACT_VALIDATION_PROFILE_NAMES", "1"):
            pv["should_promote_profiles_count"] = len(pv.get("should_promote_profiles") or []) if isinstance(pv.get("should_promote_profiles"), list) else 0
            pv["should_demote_profiles_count"] = len(pv.get("should_demote_profiles") or []) if isinstance(pv.get("should_demote_profiles"), list) else 0
            pv["should_promote_profiles"] = []
            pv["should_demote_profiles"] = []
    fa = d.get("final_arbitration") if isinstance(d.get("final_arbitration"), dict) else None
    if isinstance(fa, dict):
        pv = fa.get("post_decision_audio_validation") if isinstance(fa.get("post_decision_audio_validation"), dict) else None
        if isinstance(pv, dict) and _env_on("BUSY_REDACT_VALIDATION_PROFILE_NAMES", "1"):
            pv["should_promote_profiles_count"] = len(pv.get("should_promote_profiles") or []) if isinstance(pv.get("should_promote_profiles"), list) else 0
            pv["should_demote_profiles_count"] = len(pv.get("should_demote_profiles") or []) if isinstance(pv.get("should_demote_profiles"), list) else 0
            pv["should_promote_profiles"] = []
            pv["should_demote_profiles"] = []

    for container_key in ["role_first_dsp_contract", "mastering_intent_contract"]:
        c = d.get(container_key) if isinstance(d.get(container_key), dict) else None
        if not isinstance(c, dict):
            continue
        rm = c.get("role_map") if isinstance(c.get("role_map"), dict) else None
        c.pop("private_rulepack_runtime", None)
        c.pop("private_rulepack", None)
        if "private_rulepack_summary" not in c:
            c["private_rulepack_summary"] = {"enabled": False}
        if isinstance(rm, dict):
            snippets = rm.get("private_role_prior_snippets") if isinstance(rm.get("private_role_prior_snippets"), list) else []
            rm["private_role_prior_count"] = len(snippets)
            rm["private_role_prior_snippets"] = [
                {
                    "matched_role_count": len((x or {}).get("matched_roles") or []) if isinstance(x, dict) else 0,
                    "matched_roles": (x or {}).get("matched_roles", [])[:8] if isinstance(x, dict) else [],
                    "policy": "redacted_profile_name_role_snippet_only",
                }
                for x in snippets[:8]
            ]
            if not _env_on("BUSY_STAGE_STATE_KEEP_ROLE_SOURCES", "0"):
                rm["role_source_count_by_role"] = {
                    role: len(srcs) for role, srcs in (rm.get("role_sources") or {}).items() if isinstance(srcs, dict)
                }
                rm.pop("role_sources", None)
    d = _redact_private_evidence_blocks(d)
    d.setdefault("stage_state_privacy", {})["private_reference_payloads_redacted"] = True
    d["stage_state_privacy"]["policy"] = "public_safe_stage_state; set BUSY_PUBLIC_STAGE_STATE_SAFE=0 only for local private debugging"
    return d


def _public_safe_stage_state(data: dict[str, Any]) -> dict[str, Any]:
    if not _env_on("BUSY_PUBLIC_STAGE_STATE_SAFE", "1"):
        return data
    try:
        out = deepcopy(data)
        if isinstance(out.get("decision"), dict):
            out["decision"] = _sanitize_private_decision_payload(out.get("decision"))
        ia = out.get("input_analysis") if isinstance(out.get("input_analysis"), dict) else None
        if isinstance(ia, dict):
            for key in ["private_reference_profile_context", "private_profile_db", "private_rules", "private_bundle", "profile_mastering_priors", "private_rulepack"]:
                if key in ia:
                    ia[key] = {"redacted": True, "reason": "private_payload_hidden_in_stage_state"}
            if isinstance(ia.get("role_first_dsp_contract"), dict):
                tmp_dec = {"role_first_dsp_contract": ia.get("role_first_dsp_contract")}
                ia["role_first_dsp_contract"] = _sanitize_private_decision_payload(tmp_dec).get("role_first_dsp_contract")
            out["input_analysis"] = _redact_private_evidence_blocks(ia)
        if isinstance(out.get("analysis_sources"), dict):
            out["analysis_sources"] = _redact_private_evidence_blocks(out.get("analysis_sources"))
        out = _redact_private_evidence_blocks(out)
        out["stage_state_privacy"] = {
            "public_safe": True,
            "private_reference_payloads_redacted": True,
            "disable_env": "BUSY_PUBLIC_STAGE_STATE_SAFE=0",
        }
        return out
    except Exception:
        return data


def _write_json_file(path: str | Path, data: dict[str, Any]) -> Path:
    local = Path(path)
    local.parent.mkdir(parents=True, exist_ok=True)
    safe_data = _public_safe_stage_state(data)
    local.write_text(json.dumps(safe_data, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    return local




def _brief_get(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return default if cur is None else cur


def _brief_fmt(value: Any, digits: int = 3) -> str:
    try:
        f = float(value)
        if np.isfinite(f):
            return f"{f:.{digits}f}"
    except Exception:
        pass
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    text = str(value)
    return text if len(text) <= 180 else text[:177] + "..."


def _brief_bool(value: Any) -> str:
    return "true" if bool(value) else "false"


def _brief_list(values: Any, limit: int = 8) -> str:
    if not isinstance(values, list) or not values:
        return "-"
    out = []
    for v in values[:limit]:
        if isinstance(v, dict):
            out.append(str(v.get("reason") or v.get("code") or v.get("action") or v.get("module") or v)[:140])
        else:
            out.append(str(v)[:140])
    if len(values) > limit:
        out.append(f"...(+{len(values)-limit})")
    return ", ".join(out)


def _brief_positive(*values: Any) -> bool:
    for value in values:
        try:
            if value is not None and float(value) > 0:
                return True
        except Exception:
            continue
    return False


def _brief_first_list(*values: Any) -> list[Any]:
    for value in values:
        if isinstance(value, list) and value:
            return value
    return []


def _brief_action_details(values: Any, limit: int = 6) -> str:
    if not isinstance(values, list) or not values:
        return "-"
    out: list[str] = []
    for v in values[:limit]:
        if isinstance(v, dict):
            module = str(v.get("module") or "").strip()
            action = str(v.get("action") or v.get("source") or v.get("type") or "").strip()
            freq = v.get("freq") or v.get("frequency") or v.get("center_freq")
            gain = v.get("gain_db") or v.get("max_gr_db") or v.get("strength")
            target = str(v.get("target") or "").strip()
            pieces = [x for x in [module, action] if x]
            if freq is not None:
                try:
                    pieces.append(f"{float(freq):.0f}Hz")
                except Exception:
                    pieces.append(str(freq)[:32])
            if gain is not None:
                try:
                    label = "str" if "strength" in v and "gain_db" not in v and "max_gr_db" not in v else "dB"
                    pieces.append(f"{label}={float(gain):.3f}")
                except Exception:
                    pieces.append(str(gain)[:32])
            if target:
                pieces.append(target)
            out.append(" / ".join(pieces)[:180] if pieces else str(v)[:180])
        else:
            out.append(str(v)[:180])
    if len(values) > limit:
        out.append(f"...(+{len(values)-limit})")
    return "; ".join(out)


def _brief_effective_active(plan: dict[str, Any], render: dict[str, Any], *, include_plan_counts: bool = True) -> bool:
    plan = plan if isinstance(plan, dict) else {}
    render = render if isinstance(render, dict) else {}
    if plan.get("active") or plan.get("diagnosis_active") or render.get("active") or render.get("render_affecting"):
        return True
    count_values = [
        render.get("applied_static_move_count"), render.get("applied_dynamic_move_count"), render.get("applied_time_domain_action_count"),
        render.get("applied_move_count"), render.get("dynamic_move_count"), render.get("time_domain_action_count"),
    ]
    if include_plan_counts:
        count_values += [plan.get("static_move_count"), plan.get("dynamic_move_count"), plan.get("time_domain_action_count"), plan.get("applied_move_count")]
    return _brief_positive(*count_values)


def _debug_brief_supabase_path(args: argparse.Namespace) -> str | None:
    raw = str(getattr(args, "supabase_stage_state", "") or "").strip()
    if not raw:
        raw = str(getattr(args, "supabase_output", "") or "").strip()
    if not raw:
        return None
    base = raw.rsplit("/", 1)
    parent = base[0] if len(base) == 2 else ""
    name = "debug_brief.md"
    return f"{parent}/{name}" if parent else name


def _build_debug_brief_markdown(report: dict[str, Any], state: dict[str, Any] | None, args: argparse.Namespace | None = None) -> str:
    state = state if isinstance(state, dict) else {}
    out = report.get("output_analysis") if isinstance(report.get("output_analysis"), dict) else {}
    inp = report.get("input_analysis") if isinstance(report.get("input_analysis"), dict) else {}
    finish = report.get("commercial_loudness_finish") if isinstance(report.get("commercial_loudness_finish"), dict) else {}
    export_lock = report.get("final_export_lock") if isinstance(report.get("final_export_lock"), dict) else {}
    v57 = finish.get("v57_commercial_loudness_feasibility") if isinstance(finish.get("v57_commercial_loudness_feasibility"), dict) else report.get("v57_commercial_loudness_feasibility", {})
    v57_sel = finish.get("v57_damage_aware_safe_selection") if isinstance(finish.get("v57_damage_aware_safe_selection"), dict) else report.get("v57_damage_aware_safe_selection", {})
    v58 = report.get("v58_spectral_conditioning_plan") if isinstance(report.get("v58_spectral_conditioning_plan"), dict) else {}
    v58r = report.get("v58_spectral_conditioning_render_report") if isinstance(report.get("v58_spectral_conditioning_render_report"), dict) else {}
    v58_1 = v58.get("v58_1_report_driven_spectral_efficiency") if isinstance(v58.get("v58_1_report_driven_spectral_efficiency"), dict) else {}
    if not v58_1 and isinstance(v58r.get("v58_1_report_driven_spectral_efficiency"), dict):
        v58_1 = v58r.get("v58_1_report_driven_spectral_efficiency")
    v59 = report.get("v59_lf_crest_source_conditioning_plan") if isinstance(report.get("v59_lf_crest_source_conditioning_plan"), dict) else {}
    v59r = report.get("v59_lf_crest_source_conditioning_render_report") if isinstance(report.get("v59_lf_crest_source_conditioning_render_report"), dict) else {}
    v62_8 = report.get("v62_8_mastering_blocker_preparation") if isinstance(report.get("v62_8_mastering_blocker_preparation"), dict) else {}
    if not v62_8 and isinstance((report.get("pre_limiter_report") or {}).get("v62_8_mastering_blocker_preparation"), dict):
        v62_8 = (report.get("pre_limiter_report") or {}).get("v62_8_mastering_blocker_preparation")
    v62_8_feas = v62_8.get("commercial_loudness_feasibility") if isinstance(v62_8.get("commercial_loudness_feasibility"), dict) else {}
    v62_8_class = v62_8.get("classifier") if isinstance(v62_8.get("classifier"), dict) else {}
    v58_effective_active = _brief_effective_active(v58, v58r)
    v58_render_active = _brief_effective_active({}, v58r, include_plan_counts=False)
    v58_1_effective_active = bool(v58_1.get("active") or v58_1.get("render_affecting") or _brief_positive(v58_1.get("applied_move_count"), v58_1.get("applied_static_move_count"), v58_1.get("applied_dynamic_move_count")))
    v58_1_render_affecting = bool(v58_1.get("render_affecting") or _brief_positive(v58_1.get("applied_move_count"), v58_1.get("applied_static_move_count"), v58_1.get("applied_dynamic_move_count")))
    v59_effective_active = _brief_effective_active(v59, v59r)
    v59_render_affecting = bool(v59.get("render_affecting") or v59r.get("render_affecting") or _brief_positive(v59r.get("applied_static_move_count"), v59r.get("applied_dynamic_move_count"), v59r.get("applied_time_domain_action_count")))
    v59_actions = _brief_first_list(v59.get("actions"), v59r.get("actions"), v59r.get("time_domain_actions"), v59.get("time_domain_actions"), v59r.get("dynamic_moves"), v59.get("dynamic_moves"))
    v59_2 = v59.get("v59_2_program_dependent_dynamics_stereo_side") if isinstance(v59.get("v59_2_program_dependent_dynamics_stereo_side"), dict) else {}
    if not v59_2 and isinstance(v59r.get("v59_2_program_dependent_dynamics_stereo_side"), dict):
        v59_2 = v59r.get("v59_2_program_dependent_dynamics_stereo_side")
    v59_2_controls = v59_2.get("program_dynamics_controls", {}) if isinstance(v59_2.get("program_dynamics_controls"), dict) else {}
    v59_2_effective_active = bool(v59_2.get("active") or v59_2.get("render_affecting") or _brief_positive(v59r.get("applied_stereo_side_action_count")))
    stage_m = _get_stage_m_report_from_context(features=inp, state=state, report=report)
    stem_evidence = inp.get("stem_evidence_assist") if isinstance(inp.get("stem_evidence_assist"), dict) else {}
    v61 = report.get("v61_render_measured_feedback_optimizer") if isinstance(report.get("v61_render_measured_feedback_optimizer"), dict) else {}
    if not v61 and isinstance((report.get("pre_limiter_report") or {}).get("v61_render_measured_feedback_optimizer"), dict):
        v61 = (report.get("pre_limiter_report") or {}).get("v61_render_measured_feedback_optimizer")
    v61_passes = v61.get("passes") if isinstance(v61.get("passes"), list) else []
    v61_selected = None
    try:
        _sel_idx = int(v61.get("selected_pass", 0) or 0)
        for _p in v61_passes:
            if isinstance(_p, dict) and int(_p.get("pass_index", -1) or -1) == _sel_idx:
                v61_selected = _p
                break
    except Exception:
        v61_selected = None
    v61_selected = v61_selected if isinstance(v61_selected, dict) else {}
    v60 = report.get("v60_deterministic_multistage_limiter") if isinstance(report.get("v60_deterministic_multistage_limiter"), dict) else {}
    ownership = report.get("mastering_ownership") if isinstance(report.get("mastering_ownership"), dict) else {}
    if not ownership and isinstance(finish.get("mastering_ownership"), dict):
        ownership = finish.get("mastering_ownership") or {}
    if not ownership and isinstance(v60.get("v6350_limiter_workload_budget_plan"), dict) and isinstance((v60.get("v6350_limiter_workload_budget_plan") or {}).get("mastering_ownership"), dict):
        ownership = (v60.get("v6350_limiter_workload_budget_plan") or {}).get("mastering_ownership") or {}
    push_gate = report.get("commercial_push_safety_gate") if isinstance(report.get("commercial_push_safety_gate"), dict) else {}
    if not push_gate and isinstance(finish.get("commercial_push_safety_gate"), dict):
        push_gate = finish.get("commercial_push_safety_gate") or {}
    v6316 = report.get("v6316_mastering_chain_reconnect") if isinstance(report.get("v6316_mastering_chain_reconnect"), dict) else {}
    v60_gate = v60.get("acceptance_gate") if isinstance(v60.get("acceptance_gate"), dict) else {}
    v60_metrics = v60_gate.get("metrics") if isinstance(v60_gate.get("metrics"), dict) else {}
    v6362_contract = report.get("v6362_dml_ownership_contract") if isinstance(report.get("v6362_dml_ownership_contract"), dict) else (v60.get("v6362_dml_ownership_contract") if isinstance(v60.get("v6362_dml_ownership_contract"), dict) else (v60_gate.get("v6362_dml_ownership_contract") if isinstance(v60_gate.get("v6362_dml_ownership_contract"), dict) else {}))
    gem = report.get("final_master_highlight_review") if isinstance(report.get("final_master_highlight_review"), dict) else finish.get("final_master_highlight_review", {}) if isinstance(finish.get("final_master_highlight_review"), dict) else {}
    airqc = report.get("adaptive_input_relative_qc_governance") if isinstance(report.get("adaptive_input_relative_qc_governance"), dict) else {}
    damage_score = airqc.get("input_relative_damage_scoring") if isinstance(airqc.get("input_relative_damage_scoring"), dict) else {}
    damage_decision = damage_score.get("decision") if isinstance(damage_score.get("decision"), dict) else {}
    damage_axes = damage_score.get("axis_results") if isinstance(damage_score.get("axis_results"), list) else []
    v6400 = report.get("v6400_finishing_intelligence") if isinstance(report.get("v6400_finishing_intelligence"), dict) else report.get("finishing_intelligence") if isinstance(report.get("finishing_intelligence"), dict) else {}
    v6400_playback = v6400.get("playback_translation_auditor") if isinstance(v6400.get("playback_translation_auditor"), dict) else report.get("playback_translation_auditor") if isinstance(report.get("playback_translation_auditor"), dict) else {}
    v6400_codec = v6400.get("codec_stress_test") if isinstance(v6400.get("codec_stress_test"), dict) else report.get("codec_stress_test") if isinstance(report.get("codec_stress_test"), dict) else {}
    v6400_shadow = v6400.get("shadow_master_candidate_selection") if isinstance(v6400.get("shadow_master_candidate_selection"), dict) else report.get("shadow_master_candidate_selection") if isinstance(report.get("shadow_master_candidate_selection"), dict) else {}
    v6400_damage = v6400.get("input_relative_damage_ownership") if isinstance(v6400.get("input_relative_damage_ownership"), dict) else report.get("input_relative_damage_ownership") if isinstance(report.get("input_relative_damage_ownership"), dict) else {}
    v6400_cal = v6400.get("proxy_actual_calibration") if isinstance(v6400.get("proxy_actual_calibration"), dict) else report.get("proxy_actual_calibration") if isinstance(report.get("proxy_actual_calibration"), dict) else {}
    v6400_replay = v6400.get("decision_replay") if isinstance(v6400.get("decision_replay"), dict) else report.get("decision_replay") if isinstance(report.get("decision_replay"), dict) else {}
    v6400_reg = v6400.get("corpus_regression") if isinstance(v6400.get("corpus_regression"), dict) else report.get("corpus_regression") if isinstance(report.get("corpus_regression"), dict) else {}
    v6404 = report.get("v6404_commercial_maximum_density") if isinstance(report.get("v6404_commercial_maximum_density"), dict) else report.get("commercial_maximum_density") if isinstance(report.get("commercial_maximum_density"), dict) else {}
    v6404_selected = v6404.get("selected_candidate_id") if isinstance(v6404, dict) else None
    v6404_adv = v6404.get("damage_advisories") if isinstance(v6404.get("damage_advisories"), list) else []
    v6404_blocks = v6404.get("blocking_reasons") if isinstance(v6404.get("blocking_reasons"), list) else []
    rows = []
    rows.append("# Busy Auto Mastering Debug Brief")
    rows.append("")
    rows.append("## 1. Version / Job")
    rows.append(f"- engine_version: `{report.get('engine_version', 'n/a')}`")
    rows.append(f"- report_schema: `{report.get('mastering_report_schema_version', 'n/a')}`")
    rows.append(f"- job_id: `{state.get('job_id') or _brief_get(report, 'job_id', default='n/a')}`")
    rows.append(f"- stage_state_path: `{getattr(args, 'supabase_stage_state', '') if args is not None else ''}`")
    rows.append(f"- output_path: `{getattr(args, 'supabase_output', '') if args is not None else ''}`")
    rows.append("")
    rows.append("## 2. Final Output Metrics")
    _finish_auth = finish.get('final_measurement_authority') if isinstance(finish.get('final_measurement_authority'), dict) else {}
    _out_auth = out.get('measurement_authority') if isinstance(out.get('measurement_authority'), dict) else _brief_get(out, 'loudness', 'measurement_authority', default={})
    rows.append(f"- input_lufs: {_brief_fmt(report.get('input_lufs') or inp.get('integrated_lufs') or _brief_get(report, 'original_input_analysis_before_pre_clean', 'integrated_lufs'))}")
    rows.append(f"- final_lufs: {_brief_fmt(out.get('integrated_lufs'))}")
    rows.append(f"- final_lufs_authority: `{finish.get('final_lufs_authority') or _out_auth.get('authority') or 'n/a'}`")
    rows.append(f"- final_lufs_stereo: {_brief_fmt(finish.get('final_lufs_stereo') or out.get('stereo_integrated_lufs') or _brief_get(out, 'loudness', 'stereo_integrated_lufs'))}")
    rows.append(f"- final_lufs_mono_downmix: {_brief_fmt(finish.get('final_lufs_mono_downmix') or out.get('mono_downmix_lufs') or _brief_get(out, 'loudness', 'mono_downmix_lufs'))}")
    rows.append(f"- stereo_minus_mono_lufs_db: {_brief_fmt(_finish_auth.get('stereo_minus_mono_lufs_db') or _out_auth.get('stereo_minus_mono_lufs_db'))}")
    rows.append(f"- measurement_warning: `{_finish_auth.get('warning') or _out_auth.get('warning') or '-'}`")
    _mtg = report.get('v6314_4_mono_translation_guard') if isinstance(report.get('v6314_4_mono_translation_guard'), dict) else report.get('v6314_3_mono_translation_guard') if isinstance(report.get('v6314_3_mono_translation_guard'), dict) else finish.get('v6314_4_mono_translation_guard') if isinstance(finish.get('v6314_4_mono_translation_guard'), dict) else finish.get('v6314_3_mono_translation_guard') if isinstance(finish.get('v6314_3_mono_translation_guard'), dict) else {}
    if _mtg:
        rows.append(f"- v63.1.4.4.1 mono_translation_guard: active={_brief_bool(_mtg.get('active'))}, applied={_brief_bool(_mtg.get('applied'))}, before_delta={_brief_fmt(_mtg.get('before_stereo_minus_mono_lufs_db'))}, after_delta={_brief_fmt(_mtg.get('after_stereo_minus_mono_lufs_db'))}, trim={_brief_fmt(_mtg.get('side_trim_db'))} dB, reason=`{_mtg.get('reason') or 'n/a'}`")
    rows.append(f"- true_peak_dbtp: {_brief_fmt(out.get('approx_true_peak_dbfs'))}")
    rows.append(f"- lra_lu: {_brief_fmt(_brief_get(out, 'loudness', 'short_term_lufs', 'lra_lu', default=out.get('lra_lu')))}")
    rows.append(f"- crest_factor_db: {_brief_fmt(out.get('crest_factor_db') or _brief_get(out, 'loudness', 'crest_factor_db'))}")
    _top_pr = report.get('processing_report') if isinstance(report.get('processing_report'), dict) else {}
    _lit = _top_pr.get('loudness_iteration') if isinstance(_top_pr.get('loudness_iteration'), dict) else {}
    _tg2 = _top_pr.get('v6314_2_bamix_loudness_target_guard') if isinstance(_top_pr.get('v6314_2_bamix_loudness_target_guard'), dict) else (_lit.get('v6314_2_bamix_loudness_target_guard') if isinstance(_lit.get('v6314_2_bamix_loudness_target_guard'), dict) else {})
    rows.append(f"- preferred_target_lufs: {_brief_fmt(report.get('preferred_target_lufs') or finish.get('preferred_target_lufs') or _tg2.get('original_target_lufs') or finish.get('target_lufs'))}")
    rows.append(f"- actual_render_target_lufs: {_brief_fmt(report.get('actual_render_target_lufs') or finish.get('actual_render_target_lufs') or _tg2.get('effective_target_lufs') or _lit.get('target_lufs'))}")
    rows.append(f"- target_lufs_legacy: {_brief_fmt(finish.get('target_lufs'))}")
    rows.append(f"- commercial_floor_lufs: {_brief_fmt(finish.get('commercial_floor_lufs'))}")
    _ft44 = report.get('v6314_4_commercial_floor_tolerance') if isinstance(report.get('v6314_4_commercial_floor_tolerance'), dict) else finish.get('v6314_4_commercial_floor_tolerance') if isinstance(finish.get('v6314_4_commercial_floor_tolerance'), dict) else {}
    _ts6362 = (
        report.get('target_semantics_v6362_10') if isinstance(report.get('target_semantics_v6362_10'), dict)
        else (report.get('target_semantics_v6362_9') if isinstance(report.get('target_semantics_v6362_9'), dict)
        else (report.get('target_semantics_v6362_8') if isinstance(report.get('target_semantics_v6362_8'), dict)
        else (report.get('target_semantics_v6362_7') if isinstance(report.get('target_semantics_v6362_7'), dict)
        else (report.get('target_semantics_v6362_6') if isinstance(report.get('target_semantics_v6362_6'), dict)
        else (report.get('target_semantics_v6362_5') if isinstance(report.get('target_semantics_v6362_5'), dict)
        else (report.get('target_semantics_v6362_4') if isinstance(report.get('target_semantics_v6362_4'), dict)
        else (report.get('target_semantics_v6362_2') if isinstance(report.get('target_semantics_v6362_2'), dict) else {})))))))
    )
    rows.append(f"- target_met_legacy_floor_bool: {_brief_bool(finish.get('target_met'))}")
    if _ts6362:
        rows.append(f"- commercial_floor_met: {_brief_bool(_ts6362.get('commercial_floor_met'))}, floor_margin={_brief_fmt(_ts6362.get('commercial_floor_margin_lu'))} LU")
        rows.append(f"- target_reached: {_brief_bool(_ts6362.get('target_reached'))}, target_gap={_brief_fmt(_ts6362.get('target_lufs_gap_lu'))} LU, target_miss_allowed_by_dml={_brief_bool(_ts6362.get('target_miss_allowed_by_dml_contract'))}")
    elif report.get('target_reached') is not None or report.get('commercial_floor_met') is not None:
        rows.append(f"- commercial_floor_met: {_brief_bool(report.get('commercial_floor_met'))}, floor_margin={_brief_fmt(report.get('commercial_floor_margin_lu'))} LU")
        rows.append(f"- target_reached: {_brief_bool(report.get('target_reached'))}, target_gap={_brief_fmt(report.get('target_lufs_gap_lu'))} LU, target_miss_allowed_by_dml={_brief_bool(report.get('target_miss_allowed_by_dml', report.get('target_miss_allowed')))}")
    if _ft44:
        rows.append(f"- v63.1.4.4.1 floor_tolerance: applied={_brief_bool(_ft44.get('applied'))}, exact={_brief_bool(_ft44.get('target_met_exact'))}, within_tol={_brief_bool(_ft44.get('target_met_with_tolerance'))}, floor_margin={_brief_fmt(_ft44.get('floor_margin_lu'))}, tol={_brief_fmt(_ft44.get('tolerance_lu'))}, reason=`{_ft44.get('reason') or 'n/a'}`")
    rows.append(f"- final_source: `{report.get('final_source') or finish.get('selected_final_source') or finish.get('fallback_final_source') or report.get('selected_final_source') or 'n/a'}`")
    _fsa6362 = (
        report.get('final_source_authority_v6362_10') if isinstance(report.get('final_source_authority_v6362_10'), dict)
        else (report.get('final_source_authority_v6362_9') if isinstance(report.get('final_source_authority_v6362_9'), dict)
        else (report.get('final_source_authority_v6362_8') if isinstance(report.get('final_source_authority_v6362_8'), dict)
        else (report.get('final_source_authority_v6362_7') if isinstance(report.get('final_source_authority_v6362_7'), dict)
        else (report.get('final_source_authority_v6362_6') if isinstance(report.get('final_source_authority_v6362_6'), dict)
        else (report.get('final_source_authority_v6362_5') if isinstance(report.get('final_source_authority_v6362_5'), dict)
        else (report.get('final_source_authority_v6362_4') if isinstance(report.get('final_source_authority_v6362_4'), dict)
        else (report.get('final_source_authority_v6362_3') if isinstance(report.get('final_source_authority_v6362_3'), dict)
        else (report.get('final_source_authority_v6362_2') if isinstance(report.get('final_source_authority_v6362_2'), dict) else {}))))))))
    )
    if _fsa6362:
        rows.append(f"- final_source_authority: reason=`{_fsa6362.get('resolution_reason') or 'n/a'}`, prev=`{_fsa6362.get('previous_report_final_source') or 'n/a'}`, stale_standard={_brief_bool(_fsa6362.get('previous_report_final_source_was_stale_standard'))}, stale_v60_after_rollback={_brief_bool(_fsa6362.get('previous_report_final_source_was_stale_v60_after_rollback'))}, v60_applied={_brief_bool(_fsa6362.get('v60_applied_or_accepted'))}, rollback_standard={_brief_bool(_fsa6362.get('rollback_to_standard_final'))}, rollback_overrode_prev={_brief_bool(_fsa6362.get('previous_report_final_source_overridden_by_rollback_standard'))}")
    rows.append("")
    rows.append("## 3. Export Lock / Candidate Handoff")
    rows.append(f"- final_export_lock.accepted: {('n/a' if export_lock.get('accepted') is None else _brief_bool(export_lock.get('accepted')))}")
    rows.append(f"- comparison_analyzer: `{export_lock.get('comparison_analyzer', 'n/a')}`")
    rows.append(f"- buffer_identity_match: {('n/a' if export_lock.get('buffer_identity_match') is None else _brief_bool(export_lock.get('buffer_identity_match')))}")
    rows.append(f"- mismatch_reasons: {_brief_list(export_lock.get('mismatch_reasons'))}")
    rows.append(f"- rollback_triggered: {_brief_bool(finish.get('rollback_triggered') or export_lock.get('rollback_to_standard_final'))}")
    rows.append("")
    rows.append("## 3A. Input-Relative Damage Scoring")
    rows.append(f"- active: {_brief_bool(damage_score.get('enabled') if damage_score else airqc.get('active'))}")
    rows.append(f"- weighted_risk: {_brief_fmt(damage_score.get('weighted_risk'))}")
    rows.append(f"- decision: `{damage_decision.get('acceptance', 'n/a')}` action=`{damage_decision.get('recommended_action', 'n/a')}` warning=`{damage_decision.get('recommended_warning', '-')}`")
    _axis_parts = []
    for _axis in damage_axes[:12]:
        if not isinstance(_axis, dict):
            continue
        _val = _axis.get('value')
        if isinstance(_val, (int, float)):
            _axis_parts.append(f"{_axis.get('axis','?')}={_axis.get('class','?')}({_brief_fmt(_val)})")
        else:
            _axis_parts.append(f"{_axis.get('axis','?')}={_axis.get('class','?')}")
    rows.append(f"- axis_classes: {_brief_list(_axis_parts, limit=12)}")
    _cs = report.get('v6314_4_floor_boundary_crest_semantics') if isinstance(report.get('v6314_4_floor_boundary_crest_semantics'), dict) else report.get('v6314_3_floor_boundary_crest_semantics') if isinstance(report.get('v6314_3_floor_boundary_crest_semantics'), dict) else finish.get('v6314_4_floor_boundary_crest_semantics') if isinstance(finish.get('v6314_4_floor_boundary_crest_semantics'), dict) else finish.get('v6314_3_floor_boundary_crest_semantics') if isinstance(finish.get('v6314_3_floor_boundary_crest_semantics'), dict) else {}
    if _cs:
        rows.append(f"- v63.1.4.4.1 crest_semantics: `{_cs.get('classification','n/a')}`, premaster_loss={_brief_fmt(_cs.get('premaster_to_final_crest_loss_db'))}, max={_brief_fmt(_cs.get('max_premaster_crest_loss_db'))}, over_by={_brief_fmt(_cs.get('crest_policy_over_by_db'))}, floor_margin={_brief_fmt(_cs.get('floor_margin_lu'))}, display_warning=`{_cs.get('display_warning') or '-'}`")
    _air_lra = airqc.get('lra') if isinstance(airqc.get('lra'), dict) else {}
    _air_crest = airqc.get('crest') if isinstance(airqc.get('crest'), dict) else {}
    _air_tp = airqc.get('true_peak_pressure') if isinstance(airqc.get('true_peak_pressure'), dict) else {}
    rows.append(f"- LRA input/final/loss/class: {_brief_fmt(_air_lra.get('input_lra_lu'))} / {_brief_fmt(_air_lra.get('final_lra_lu'))} / {_brief_fmt(_air_lra.get('lra_loss_lu'))} / `{_air_lra.get('classification','n/a')}`")
    rows.append(f"- Crest input/final/loss/class: {_brief_fmt(_air_crest.get('input_crest_db'))} / {_brief_fmt(_air_crest.get('final_crest_db'))} / {_brief_fmt(_air_crest.get('crest_loss_db'))} / `{_air_crest.get('classification','n/a')}`")
    rows.append(f"- TP input/final/delta/class: {_brief_fmt(_air_tp.get('input_true_peak_dbtp'))} / {_brief_fmt(_air_tp.get('final_true_peak_dbtp'))} / {_brief_fmt(_air_tp.get('true_peak_pressure_delta_db'))} / `{_air_tp.get('classification','n/a')}`")
    rows.append(f"- final_warnings: {_brief_list(report.get('remaining_safety_warnings'))}")
    rows.append("")
    rows.append("## 4. Stem Evidence Assist / Stage M Replacement")
    rows.append(f"- stem_evidence.enabled/provided/usable: {_brief_bool(stem_evidence.get('enabled'))} / {_brief_bool(stem_evidence.get('provided'))} / {_brief_bool(stem_evidence.get('usable'))}")
    rows.append(f"- stem_evidence.global_reliability: {_brief_fmt(stem_evidence.get('global_reliability'))}")
    _sea_res = stem_evidence.get('residue_witness') if isinstance(stem_evidence.get('residue_witness'), dict) else {}
    _sea_prot = stem_evidence.get('protection_priorities') if isinstance(stem_evidence.get('protection_priorities'), dict) else {}
    rows.append(f"- stem_evidence.residue: side_high_hash={_brief_fmt(_sea_res.get('side_high_hash_risk'))}, synthetic_air={_brief_fmt(_sea_res.get('synthetic_air_risk'))}, low_mid_mud={_brief_fmt(_sea_res.get('low_mid_ai_mud_risk'))}")
    rows.append(f"- stem_evidence.protection: vocal_presence={_brief_fmt(_sea_prot.get('vocal_presence'))}, bass={_brief_fmt(_sea_prot.get('bass_fundamental'))}, kick={_brief_fmt(_sea_prot.get('kick_transient'))}, hats={_brief_fmt(_sea_prot.get('drum_hat_transients'))}")
    rows.append(f"- stem_evidence.top_findings: {_brief_list(stem_evidence.get('top_findings'))}")
    rows.append(f"- stage_m.replaced_by_stem_evidence_assist: {_brief_bool(True)}")
    rows.append("")
    rows.append("## 4b. Stage M Legacy Gate")
    rows.append(f"- stage_m.enabled: {_brief_bool(stage_m.get('enabled'))}")
    rows.append(f"- stage_m.attempted: {_brief_bool(stage_m.get('attempted'))}")
    rows.append(f"- stage_m.mode: `{stage_m.get('mode', 'n/a')}`")
    rows.append(f"- stage_m.selected_source: `{stage_m.get('selected_source', 'n/a')}`")
    rows.append(f"- stage_m.use_premaster_for_mastering: {_brief_bool(stage_m.get('use_premaster_for_mastering') or stage_m.get('used_for_mastering'))}")
    rows.append(f"- stage_m.reason: `{stage_m.get('reason', 'n/a')}`")
    rows.append(f"- stage_m.fallback_triggers: {_brief_list(stage_m.get('fallback_triggers'))}")
    rows.append(f"- stage_m.global_reconstruction_score/grade: {_brief_fmt(stage_m.get('global_reconstruction_score'),0)} / `{stage_m.get('global_reconstruction_grade', 'n/a')}`")
    rows.append(f"- stage_m.reconstruction_policy: `{stage_m.get('reconstruction_policy', 'n/a')}`")
    rows.append(f"- stage_m.track_count/usable/trust_usable/mixdown_candidates: {_brief_fmt(stage_m.get('track_count_analyzed'),0)} / {_brief_fmt(stage_m.get('usable_track_count'),0)} / {_brief_fmt(stage_m.get('trust_usable_track_count'),0)} / {_brief_fmt(stage_m.get('stage_m_mixdown_candidate_count'),0)}")
    rows.append(f"- stage_m.avg_track_trust_score: {_brief_fmt(stage_m.get('avg_track_trust_score'))}")
    rows.append(f"- stage_m.diagnostics: {_brief_list(stage_m.get('diagnostics'))}")
    rows.append(f"- stage_m.downstream_stem_assist_bypassed: {_brief_bool(stage_m.get('downstream_stem_assist_bypassed'))}, reason=`{stage_m.get('downstream_stem_assist_bypass_reason', 'n/a')}`")
    _mix = stage_m.get('stage_m_mixdown_candidate') if isinstance(stage_m.get('stage_m_mixdown_candidate'), dict) else {}
    rows.append(f"- stage_m.mixdown_candidate: accepted={_brief_bool(_mix.get('accepted'))}, score={_brief_fmt(_mix.get('score'))}, reason=`{_mix.get('reason', 'n/a')}`, validation=`{_mix.get('validation_status', 'n/a')}`, blockers={_brief_list(_mix.get('blockers'))}")
    _builder = stage_m.get('stage_m_builder_summary') if isinstance(stage_m.get('stage_m_builder_summary'), dict) else {}
    rows.append(f"- stage_m.builder: accepted={_brief_bool(_builder.get('accepted'))}, reason=`{_builder.get('reason', 'n/a')}`, usable_stems={_brief_fmt(_builder.get('usable_stem_count'),0)}, audio_contributors={_brief_fmt(_builder.get('audio_contribution_stem_count'),0)}, trust_usable={_brief_fmt(_builder.get('trust_usable_stem_count'),0)}, avg_reliability={_brief_fmt(_builder.get('avg_reliability_score'))}, gain_solved_recon_error_db={_brief_fmt(_builder.get('gain_solved_reconstruction_error_db'))}")
    _srg = _builder.get('stem_restoration_gate') if isinstance(_builder.get('stem_restoration_gate'), dict) else {}
    if _srg:
        _modes = _srg.get('repair_mode_counts') if isinstance(_srg.get('repair_mode_counts'), dict) else {}
        rows.append(f"- stage_m.stem_restoration_gate: enabled={_brief_bool(_srg.get('enabled'))}, attempted={_brief_bool(_srg.get('attempted'))}, audio_contributors={_brief_fmt(_srg.get('audio_contribution_count'),0)}, clean_preserve={_brief_fmt(_modes.get('clean_preserve_audio'),0)}, repair_limited={_brief_fmt(_modes.get('repair_limited_audio'),0)}, repair_intensive={_brief_fmt(_modes.get('repair_intensive_audio'),0)}, protected_passthrough={_brief_fmt(_modes.get('protected_passthrough_audio'),0)}, metadata_only={_brief_fmt(_modes.get('metadata_only'),0)}, cleaned={_brief_fmt(_srg.get('cleaned_count'),0)}, avg_artifact_risk={_brief_fmt(_srg.get('avg_artifact_risk'))}")
    _fae = _builder.get('full_audio_contribution_enforcement') if isinstance(_builder.get('full_audio_contribution_enforcement'), dict) else {}
    if _fae:
        rows.append(f"- stage_m.full_audio_contribution_enforcement: enabled={_brief_bool(_fae.get('enabled'))}, non_silent={_brief_fmt(_fae.get('non_silent_count'),0)}, forced={_brief_fmt(_fae.get('forced_audio_contribution_count'),0)}, legacy_rewrites={_brief_fmt(_fae.get('legacy_permission_rewrite_count'),0)}, metadata_only={_brief_fmt(_fae.get('metadata_only_count'),0)}, policy=`{_fae.get('policy','n/a')}`")
    _premix = _builder.get('premix_feasibility_gate') if isinstance(_builder.get('premix_feasibility_gate'), dict) else {}
    rows.append(f"- stage_m.premix_gate: passed={_brief_bool(_premix.get('passed'))}, score={_brief_fmt(_premix.get('score'))}, blockers={_brief_list(_premix.get('blockers'))}")
    _engine = _builder.get('stage_m_mixing_engine') if isinstance(_builder.get('stage_m_mixing_engine'), dict) else {}
    _engsel = _engine.get('selected') if isinstance(_engine.get('selected'), dict) else {}
    rows.append(f"- stage_m.mixing_engine: schema=`{_engine.get('schema_version', 'n/a')}`, selected_pass={_brief_fmt(_engine.get('selected_pass'),0)}, executed_pass_count={_brief_fmt(_engine.get('executed_pass_count'),0)}, accepted_pass_count={_brief_fmt(_engine.get('accepted_pass_count'),0)}, score={_brief_fmt(_engsel.get('score'))}, blockers={_brief_list(_engsel.get('blockers'))}")
    _proxy = _engine.get('proxy_window_summary') if isinstance(_engine.get('proxy_window_summary'), dict) else {}
    _fullval = _engine.get('final_full_validation') if isinstance(_engine.get('final_full_validation'), dict) else {}
    if _engine.get('proxy_measured_feedback_loop') is not None or _proxy or _fullval:
        rows.append(f"- stage_m.proxy_feedback: active={_brief_bool(_engine.get('proxy_measured_feedback_loop'))}, windows={_brief_fmt(_proxy.get('window_count'),0)}, proxy_sec={_brief_fmt(_proxy.get('total_proxy_sec'))}, final_full_score={_brief_fmt(_fullval.get('full_score'))}, final_full_blockers={_brief_list(_fullval.get('full_blockers'))}, policy=`{_fullval.get('policy', 'n/a')}`")
    _hist = _engine.get('candidate_history') if isinstance(_engine.get('candidate_history'), list) else []
    if _hist:
        _parts = []
        for _h in _hist[:6]:
            _fu = _h.get('feedback_update') if isinstance(_h.get('feedback_update'), dict) else {}
            _uc = _fu.get('update_count', len(_fu.get('updates') or []))
            _bc = _fu.get('bridge_update_count', len(_fu.get('bridge_updates') or []))
            _mab = _fu.get('micro_ab') if isinstance(_fu.get('micro_ab'), dict) else {}
            _mab_sel = _fu.get('micro_ab_selected_family_count', _mab.get('selected_family_count'))
            _mab_ben = _fu.get('micro_ab_beneficial_family_count', _mab.get('beneficial_family_count'))
            _mab_dec = _fu.get('micro_ab_decision', _mab.get('decision', 'n/a'))
            _parts.append(f"p{_brief_fmt(_h.get('pass_index'),0)}:{_h.get('status','n/a')}/score={_brief_fmt(_h.get('score'))}/mode={_h.get('keep_or_rollback','n/a')}/updates={_brief_fmt(_uc,0)}/bridge={_brief_fmt(_bc,0)}/microAB={_brief_fmt(_mab_sel,0)}/{_brief_fmt(_mab_ben,0)}/{_mab_dec}")
        rows.append(f"- stage_m.proxy_passes: {'; '.join(_parts)}")
    _mab_engine = _engine.get('micro_ab_action_selector') if isinstance(_engine.get('micro_ab_action_selector'), dict) else {}
    if _mab_engine:
        rows.append(f"- stage_m.micro_ab_selector: enabled={_brief_bool(_mab_engine.get('enabled'))}, family_tests={_brief_fmt(_mab_engine.get('family_test_count'),0)}, selected_families={_brief_fmt(_mab_engine.get('selected_family_count'),0)}, beneficial_families={_brief_fmt(_mab_engine.get('beneficial_family_count'),0)}, policy=`{_mab_engine.get('policy','n/a')}`")
    _plan_count = _builder.get('processing_plan_count')
    _cross_count = _builder.get('cross_stem_action_count')
    _tz_count = _builder.get('target_zone_action_count')
    _fr_count = _builder.get('full_role_mix_action_count')
    _rdsp_count = _builder.get('role_dsp_primitive_action_count')
    _adsp_count = _builder.get('advanced_dsp_module_action_count')
    rows.append(f"- stage_m.processing_plan: stem_moves={_brief_fmt(_plan_count,0)}, cross_stem_actions={_brief_fmt(_cross_count,0)}, target_zone_actions={_brief_fmt(_tz_count,0)}, full_role_actions={_brief_fmt(_fr_count,0)}, role_dsp_actions={_brief_fmt(_rdsp_count,0)}, advanced_dsp_actions={_brief_fmt(_adsp_count,0)}")
    _tzr = _builder.get('target_zone_resolver') if isinstance(_builder.get('target_zone_resolver'), dict) else {}
    if _tzr:
        _before = _tzr.get('before') if isinstance(_tzr.get('before'), dict) else {}
        _after = _tzr.get('after') if isinstance(_tzr.get('after'), dict) else {}
        rows.append(f"- stage_m.target_zones: active={_brief_bool(_tzr.get('active'))}, actions={_brief_fmt(_tzr.get('action_count'),0)}, before_zone={_brief_fmt(_before.get('spectral_zone_score'))}, after_zone={_brief_fmt(_after.get('spectral_zone_score'))}, before_audibility={_brief_fmt(_before.get('role_audibility_score'))}, after_audibility={_brief_fmt(_after.get('role_audibility_score'))}, before_masking_debt={_brief_fmt(_before.get('masking_debt'))}, after_masking_debt={_brief_fmt(_after.get('masking_debt'))}")
    _frm = _builder.get('full_role_mix_rebuild') if isinstance(_builder.get('full_role_mix_rebuild'), dict) else {}
    if _frm:
        _before_fr = _frm.get('before') if isinstance(_frm.get('before'), dict) else {}
        _after_fr = _frm.get('after') if isinstance(_frm.get('after'), dict) else {}
        rows.append(f"- stage_m.full_role_mix: active={_brief_bool(_frm.get('active'))}, actions={_brief_fmt(_frm.get('action_count'),0)}, before_role_balance={_brief_fmt(_before_fr.get('role_balance_score'))}, after_role_balance={_brief_fmt(_after_fr.get('role_balance_score'))}, policy=`{_frm.get('policy','n/a')}`")
    _rdsp = _builder.get('role_dsp_primitives') if isinstance(_builder.get('role_dsp_primitives'), dict) else {}
    if _rdsp:
        _before_rd = _rdsp.get('before') if isinstance(_rdsp.get('before'), dict) else {}
        _after_rd = _rdsp.get('after') if isinstance(_rdsp.get('after'), dict) else {}
        rows.append(f"- stage_m.role_dsp_primitives: active={_brief_bool(_rdsp.get('active'))}, actions={_brief_fmt(_rdsp.get('action_count'),0)}, before_role_balance={_brief_fmt(_before_rd.get('role_balance_score'))}, after_role_balance={_brief_fmt(_after_rd.get('role_balance_score'))}, policy=`{_rdsp.get('policy','n/a')}`")
    _pmr = _builder.get('professional_mix_reconstruction') if isinstance(_builder.get('professional_mix_reconstruction'), dict) else {}
    if _pmr:
        _res = _pmr.get('residual_analysis') if isinstance(_pmr.get('residual_analysis'), dict) else {}
        _bus_exp = _pmr.get('professional_bus_candidate_expansion') if isinstance(_pmr.get('professional_bus_candidate_expansion'), dict) else {}
        rows.append(f"- stage_m.professional_mix_reconstruction: enabled={_brief_bool(_pmr.get('enabled'))}, attempted={_brief_bool(_pmr.get('attempted'))}, active={_brief_bool(_pmr.get('active'))}, selected=`{_pmr.get('selected_candidate','n/a')}`, score={_brief_fmt(_pmr.get('selected_score'))}, candidates={_brief_fmt(_pmr.get('candidate_count'),0)}, bus_candidates={_brief_fmt(_pmr.get('bus_candidate_count') or _bus_exp.get('candidate_count'),0)}, completeness={_brief_fmt(_pmr.get('mix_completeness_score'))}, residual_musical={_brief_bool(_res.get('is_musical'))}, residual_blend={_brief_fmt(_pmr.get('recommended_residual_blend'))}, policy=`{_bus_exp.get('policy') or _pmr.get('policy','n/a')}`")
        _rdc = _pmr.get('residual_decomposition_core') if isinstance(_pmr.get('residual_decomposition_core'), dict) else {}
        if _rdc:
            _rs = _rdc.get('restoration_summary') if isinstance(_rdc.get('restoration_summary'), dict) else {}
            rows.append(f"- stage_m.residual_decomposition_core: enabled={_brief_bool(_rdc.get('enabled'))}, attempted={_brief_bool(_rdc.get('attempted'))}, active={_brief_bool(_rdc.get('active'))}, apply_enabled={_brief_bool(_rdc.get('apply_enabled'))}, components={_brief_fmt(_rdc.get('component_count'),0)}, selected_components={_brief_fmt(_rdc.get('selected_component_count'),0)}, hard_rejects={_brief_fmt(_rdc.get('hard_reject_count'),0)}, selected_types=`{_brief_list(_rdc.get('component_types_selected') if isinstance(_rdc.get('component_types_selected'), list) else [])}`, reason=`{_rdc.get('reason','n/a')}`")
    elif _builder:
        rows.append("- stage_m.professional_mix_reconstruction: enabled=n/a, attempted=false, active=false, selected=`n/a`, score=n/a, candidates=n/a, completeness=n/a, residual_musical=n/a, residual_blend=n/a, reason=`missing_from_stage_m_builder_telemetry`")
    _gsr = _builder.get('gemini_stem_artifact_mix_intent_review') if isinstance(_builder.get('gemini_stem_artifact_mix_intent_review'), dict) else {}
    if _gsr:
        _pkg = _gsr.get('clip_package_summary') if isinstance(_gsr.get('clip_package_summary'), dict) else {}
        _whole = _gsr.get('whole_mix_intent_summary') if isinstance(_gsr.get('whole_mix_intent_summary'), dict) else {}
        _priors = _whole.get('priority_candidate_families') if isinstance(_whole.get('priority_candidate_families'), list) else []
        rows.append(f"- stage_m.gemini_stem_reviewer: enabled={_brief_bool(_gsr.get('enabled'))}, attempted={_brief_bool(_gsr.get('attempted'))}, available={_brief_bool(_gsr.get('available'))}, mode=`{_gsr.get('mode','n/a')}`, clips={_brief_fmt(_pkg.get('clip_count'),0)}, context_tier={_brief_fmt(_pkg.get('context_tier'),0)}, package_sec={_brief_fmt(_pkg.get('duration_sec'))}, weak_prior_only={_brief_bool((_gsr.get('fusion_policy') or {}).get('gemini_is_weak_prior_only') if isinstance(_gsr.get('fusion_policy'), dict) else True)}, priority_families=`{_brief_list(_priors)}`, reason=`{_gsr.get('reason','-')}`")
    elif _builder:
        rows.append("- stage_m.gemini_stem_reviewer: enabled=false, attempted=false, available=false, mode=`n/a`, clips=n/a, reason=`missing_or_disabled`")
    _adsp = _builder.get('advanced_dsp_modules') if isinstance(_builder.get('advanced_dsp_modules'), dict) else {}
    if _adsp:
        _before_ad = _adsp.get('before') if isinstance(_adsp.get('before'), dict) else {}
        _after_ad = _adsp.get('after') if isinstance(_adsp.get('after'), dict) else {}
        rows.append(f"- stage_m.advanced_dsp_modules: active={_brief_bool(_adsp.get('active'))}, actions={_brief_fmt(_adsp.get('action_count'),0)}, families=`{_brief_list(list((_adsp.get('family_counts') or {}).keys()) if isinstance(_adsp.get('family_counts'), dict) else [])}`, before_role_balance={_brief_fmt(_before_ad.get('role_balance_score'))}, after_role_balance={_brief_fmt(_after_ad.get('role_balance_score'))}, policy=`{_adsp.get('policy','n/a')}`")
    _e2a_count = _builder.get('evidence_to_action_count')
    _prio = _builder.get('target_priority_map') if isinstance(_builder.get('target_priority_map'), dict) else {}
    if _e2a_count is not None or _prio:
        _prio_compact = ', '.join([f"{k}={_brief_fmt(v)}" for k, v in list(_prio.items())[:7]]) if isinstance(_prio, dict) else ''
        rows.append(f"- stage_m.evidence_to_action: evidence_count={_brief_fmt(_e2a_count,0)}, priorities=`{_prio_compact or 'n/a'}`")
    _safety = _builder.get('action_specific_safety_criteria') or _engine.get('action_specific_safety_criteria')
    if isinstance(_safety, dict):
        _tp = _safety.get('true_peak') if isinstance(_safety.get('true_peak'), dict) else {}
        _fr = _safety.get('frontness') if isinstance(_safety.get('frontness'), dict) else {}
        rows.append(f"- stage_m.safety_criteria: schema=`{_safety.get('schema_version','n/a')}`, master_tp_safe={_brief_fmt(_tp.get('master_safe_dbtp'))}, master_tp_reject={_brief_fmt(_tp.get('master_reject_dbtp'))}, frontness_range={_brief_fmt(_fr.get('frontness_min'))}~{_brief_fmt(_fr.get('frontness_max'))}, policy=`hard_stop_soft_scale_reroute_quality_guard`")
    _ssum = _builder.get('safety_summary') if isinstance(_builder.get('safety_summary'), dict) else {}
    if _ssum:
        rows.append(f"- stage_m.safety_summary: hard={_brief_fmt(_ssum.get('hard_count'),0)}, soft={_brief_fmt(_ssum.get('soft_count'),0)}, reroute={_brief_fmt(_ssum.get('reroute_count'),0)}, quality={_brief_fmt(_ssum.get('quality_count'),0)}")
    _refp = _builder.get('reference_intent_profile') if isinstance(_builder.get('reference_intent_profile'), dict) else {}
    rows.append(f"- stage_m.reference_intent: reused_existing_analysis={_brief_bool(_refp.get('existing_analysis_reused'))}, source=`{_refp.get('source', 'n/a')}`")
    _roles = stage_m.get('stem_roles_compact') if isinstance(stage_m.get('stem_roles_compact'), list) else []
    if _roles:
        rows.append("- stage_m.roles: " + "; ".join([f"{str(r.get('file',''))[:32]}→{r.get('role','?')}/{r.get('tier','?')}/{_brief_fmt(r.get('reliability_score'))}" for r in _roles[:12] if isinstance(r, dict)]))
    rows.append("")
    if v62_8:
        rows.append("## 5. v62.8 Mastering Blocker Preparation")
        rows.append(f"- v62.8.active: {_brief_bool(v62_8.get('active'))}, enabled={_brief_bool(v62_8.get('enabled'))}, actions={_brief_fmt(v62_8.get('action_count'),0)}")
        rows.append(f"- v62.8.classifier.axes: {_brief_list(v62_8_class.get('active_axes'))}")
        rows.append(f"- v62.8.classifier.classes: {_brief_list(v62_8_class.get('pressure_classes'))}")
        rows.append(f"- v62.8.safe_push_budget_db: {_brief_fmt(v62_8_feas.get('safe_push_budget_db'))}, useful={_brief_bool(v62_8_feas.get('useful_safe_push_available'))}, decision=`{v62_8_feas.get('decision','n/a')}`")
        rows.append(f"- v62.8.feasibility_reasons: {_brief_list(v62_8_feas.get('reasons'))}")
        rows.append(f"- v62.8.micro_ab_families: {_brief_list(v62_8.get('micro_ab_candidate_families'))}")
        rows.append(f"- v62.8.policy: `{v62_8.get('policy','n/a')}`")
        rows.append("")
    rows.append("## 6. v57 Damage / Safe Push")
    rows.append(f"- safe_push_budget_db: {_brief_fmt(v57.get('safe_push_budget_db'))}")
    rows.append(f"- useful_safe_push_available: {_brief_bool(v57.get('useful_safe_push_available'))}")
    rows.append(f"- feasibility_decision: `{v57.get('decision', 'n/a')}`")
    rows.append(f"- feasibility_reasons: {_brief_list(v57.get('reasons') or v57.get('reason_codes') or ([v57.get('reason')] if v57.get('reason') else []))}")
    rows.append(f"- delivered_final_has_damage: {_brief_bool(v57_sel.get('delivered_final_has_damage'))}")
    rows.append(f"- final_grade_rejected: {_brief_bool(v57_sel.get('final_grade_rejected'))}")
    rows.append(f"- rejected_candidate_count: {_brief_fmt(v57_sel.get('rejected_candidate_count'), 0)}")
    rows.append("")
    rows.append("## 7. v58 / v58.1 Spectral Conditioning")
    rows.append(f"- v58.active: {_brief_bool(v58_effective_active)}")
    rows.append(f"- v58.render.active: {_brief_bool(v58_render_active)}")
    rows.append(f"- v58 static/dynamic moves: {_brief_fmt(v58.get('static_move_count') or v58r.get('static_move_count') or v58r.get('applied_static_move_count'),0)} / {_brief_fmt(v58.get('dynamic_move_count') or v58r.get('dynamic_move_count') or v58r.get('applied_dynamic_move_count'),0)}")
    rows.append(f"- v58.1.active: {_brief_bool(v58_1_effective_active)}")
    rows.append(f"- v58.1.render_affecting: {_brief_bool(v58_1_render_affecting)}")
    rows.append(f"- v58.1 applied moves: {_brief_fmt(v58_1.get('applied_move_count') or ((_brief_positive(v58_1.get('applied_static_move_count')) and v58_1.get('applied_static_move_count') or 0) + (_brief_positive(v58_1.get('applied_dynamic_move_count')) and v58_1.get('applied_dynamic_move_count') or 0)),0)}")
    rows.append(f"- v58.1 actions: {_brief_action_details(v58_1.get('actions'))}")
    rows.append("")
    rows.append("## 8. v59 LF / Crest / Rogue-Peak Conditioning")
    rows.append(f"- v59.active: {_brief_bool(v59_effective_active)}")
    rows.append(f"- v59.render_affecting: {_brief_bool(v59_render_affecting)}")
    rows.append(f"- v59 dynamic/time actions: {_brief_fmt(v59.get('dynamic_move_count') or v59r.get('dynamic_move_count'),0)} / {_brief_fmt(v59.get('time_domain_action_count') or v59r.get('time_domain_action_count'),0)}")
    rows.append(f"- v59 applied dynamic/time actions: {_brief_fmt(v59r.get('applied_dynamic_move_count'),0)} / {_brief_fmt(v59r.get('applied_time_domain_action_count'),0)}")
    rows.append(f"- v59 reasons: {_brief_list(v59.get('reasons') or v59r.get('reasons'))}")
    rows.append(f"- v59 actions: {_brief_action_details(v59_actions)}")
    rows.append(f"- v59.2.active: {_brief_bool(v59_2_effective_active)}")
    rows.append(f"- v59.2 dynamics scales: MBC {_brief_fmt(v59_2_controls.get('multiband_amount_scale'))} / DynEQ {_brief_fmt(v59_2_controls.get('dynamic_eq_intensity_scale'))} / Upward {_brief_fmt(v59_2_controls.get('upward_density_scale'))}")
    rows.append(f"- v59.2 stereo-side applied actions: {_brief_fmt(v59r.get('applied_stereo_side_action_count'),0)}")
    rows.append(f"- v59.2 actions: {_brief_action_details(v59_2.get('actions'))}")
    rows.append("")
    rows.append("## 9. v61.2 Render-Measured Feedback Optimizer")
    rows.append(f"- v61.enabled: {_brief_bool(v61.get('enabled'))}")
    rows.append(f"- v61.attempted: {_brief_bool(v61.get('attempted'))}")
    rows.append(f"- v61.active: {_brief_bool(v61.get('active'))}")
    rows.append(f"- v61.selected_pass: {_brief_fmt(v61.get('selected_pass'),0)}")
    rows.append(f"- v61.accepted_pass_count: {_brief_fmt(v61.get('accepted_pass_count'),0)}")
    rows.append(f"- v61.rollback_count: {_brief_fmt(v61.get('rollback_count'),0)}")
    rows.append(f"- v61.selected_score/base_score: {_brief_fmt(v61.get('selected_score'))} / {_brief_fmt(v61.get('base_score'))}")
    rows.append(f"- v61.score_delta: {_brief_fmt(v61.get('score_delta'))}")
    rows.append(f"- v61.stop_reason: `{v61.get('stop_reason', 'n/a')}`")
    rows.append(f"- v61.max_passes / epsilon / min_trust: {_brief_fmt(v61.get('max_passes'),0)} / {_brief_fmt(v61.get('epsilon'))} / {_brief_fmt(v61.get('min_trust_scale'))}")
    rows.append(f"- v61.selected_status: `{v61_selected.get('status', 'n/a')}`")
    rows.append(f"- v61.remaining_blockers: {_brief_list([b.get('axis') for b in (v61.get('selected_blockers_remaining') or []) if isinstance(b, dict)])}")
    if v61_passes:
        rows.append("- v61.pass_details:")
        for _p in v61_passes[:8]:
            if not isinstance(_p, dict):
                continue
            _idx = _brief_fmt(_p.get('pass_index'), 0)
            _status = str(_p.get('status', 'n/a'))
            _score = _brief_fmt(_p.get('score'))
            _delta = _brief_fmt(_p.get('score_delta_vs_best'))
            _blockers = _brief_list([b.get('axis') for b in (_p.get('blockers_detected') or []) if isinstance(b, dict)])
            _upd = _p.get('applied_update') if isinstance(_p.get('applied_update'), dict) else {}
            _acts = _brief_list([a.get('axis') for a in (_upd.get('actions') or []) if isinstance(a, dict)])
            _reasons = _brief_list(_p.get('decision_reasons') or _p.get('reject_reasons') or _p.get('hard_reject_reasons'))
            rows.append(f"  - pass {_idx}: `{_status}` score={_score} delta={_delta} blockers={_blockers} actions={_acts} reasons={_reasons}")
    rows.append("")
    rows.append("## 10. v63.5.1 Mastering Ownership / Push Safety")
    if ownership:
        rows.append(f"- ownership.class: `{ownership.get('selected_loudness_class', 'n/a')}`")
        rows.append(f"- ownership.push_authority: `{ownership.get('push_authority', 'n/a')}`")
        rows.append(f"- ownership.final_push_budget_db: {_brief_fmt(ownership.get('final_push_budget_db'))}")
        rows.append(f"- ownership.limiter_workload_budget_lu: {_brief_fmt(ownership.get('limiter_workload_budget_lu'))}")
        rows.append(f"- ownership.hard_stop_reason: `{ownership.get('hard_stop_reason') or '-'}`")
        rows.append(f"- ownership.fallback_reason: `{ownership.get('ownership_fallback_reason') or '-'}`")
        _tp_pol = ownership.get('tp_ceiling_policy') if isinstance(ownership.get('tp_ceiling_policy'), dict) else {}
        rows.append(f"- ownership.tp_policy: ceiling={_brief_fmt(_tp_pol.get('ceiling_dbtp'))} dBTP, room={_brief_fmt(_tp_pol.get('room_to_ceiling_db'))} dB, mode=`{_tp_pol.get('mode','n/a')}`")
    else:
        rows.append("- ownership: not_available")
    if push_gate:
        rows.append(f"- push_gate: allowed={_brief_bool(push_gate.get('push_allowed'))}, limited={_brief_bool(push_gate.get('push_limited'))}, artifact_fallback={_brief_bool(push_gate.get('artifact_protected_fallback'))}")
    rows.append("")
    rows.append("## 11. v60 Deterministic Multistage Limiter")
    rows.append(f"- v60.active: {_brief_bool(v60.get('active'))}")
    rows.append(f"- v60.accepted: {_brief_bool(v60.get('accepted'))}")
    rows.append(f"- v60.applied: {_brief_bool(v60.get('applied'))}")
    rows.append(f"- v60.reason: `{v60.get('reason', 'n/a')}`")
    rows.append(f"- v60.stage_count: {_brief_fmt(v60.get('stage_count'),0)}")
    rows.append(f"- v60.lufs_delta_db: {_brief_fmt(v60_metrics.get('lufs_delta_db'))}")
    rows.append(f"- v60.lra_delta_lu: {_brief_fmt(v60_metrics.get('lra_delta_lu'))}")
    rows.append(f"- v60.crest_delta_db: {_brief_fmt(v60_metrics.get('crest_delta_db'))}")
    rows.append(f"- v60.reject_reasons: {_brief_list(v60_gate.get('reasons'))}")
    if v6362_contract:
        _cmet = v6362_contract.get('metrics') if isinstance(v6362_contract.get('metrics'), dict) else {}
        rows.append(f"- v63.6.2 DML contract: active={_brief_bool(v6362_contract.get('active'))}, passed={_brief_bool(v6362_contract.get('contract_passed'))}, target_miss_allowed={_brief_bool(v6362_contract.get('target_miss_allowed'))}, push_authority=`{v6362_contract.get('push_authority','n/a')}`, allowed_push={_brief_fmt(v6362_contract.get('allowed_final_push_db'))} dB")
        rows.append(f"- v63.6.2 DML LRA: standard={_brief_fmt(_cmet.get('standard_lra_lu'))}, candidate={_brief_fmt(_cmet.get('candidate_lra_lu'))}, delta={_brief_fmt(_cmet.get('dml_lra_delta_lu'))}, authority=`{_cmet.get('candidate_lra_authority') or _cmet.get('standard_lra_authority') or 'n/a'}`")
        rows.append(f"- v63.6.2 DML target/workload: target_gap={_brief_fmt(v6362_contract.get('target_gap_after_dml_lu'))}, workload_excess={_brief_fmt(v6362_contract.get('workload_excess_over_cap_lu'))}, reject_reasons={_brief_list(v6362_contract.get('reject_reasons'))}")
    _v60_align = v60_gate.get('deep_research_alignment') if isinstance(v60_gate.get('deep_research_alignment'), dict) else {}
    if _v60_align:
        rows.append(f"- v60.deep_research_alignment: active={_brief_bool(_v60_align.get('active'))}, notes={_brief_list(_v60_align.get('governance_notes'))}")
    if v6316:
        _handoff = v6316.get('pre_limiter_handoff') if isinstance(v6316.get('pre_limiter_handoff'), dict) else {}
        rows.append("")
        rows.append("## 10B. v63.1.6.2.1 Deep Research Alignment Runtime Hotfix")
        rows.append(f"- reconnect: active={_brief_bool(v6316.get('active'))}, returned_checkpoint=`{_handoff.get('returned_checkpoint') or 'n/a'}`, pre_limiter_lufs={_brief_fmt(_handoff.get('returned_lufs'))}, limited_preview_lufs={_brief_fmt(_handoff.get('limited_preview_lufs'))}")
        rows.append(f"- commercial_finish: selected=`{v6316.get('commercial_finish_selected') or 'n/a'}`, evaluated={_brief_bool(v6316.get('commercial_finish_evaluated'))}; v60 active/applied={_brief_bool(v6316.get('v60_active'))}/{_brief_bool(v6316.get('v60_applied'))}; v61 attempted={_brief_bool(v6316.get('v61_attempted'))}")
        _floor_rec = finish.get('v63162_commercial_floor_recovery') if isinstance(finish.get('v63162_commercial_floor_recovery'), dict) else {}
        if _floor_rec:
            rows.append(f"- commercial_floor_recovery: active={_brief_bool(_floor_rec.get('active'))}, accepted={_brief_bool(_floor_rec.get('accepted'))}, reason=`{_floor_rec.get('reason','n/a')}`, needed_lift={_brief_fmt(_floor_rec.get('needed_lift_db'))}")
        _deliv_ref = finish.get('delivered_final_runtime_reference_analysis') if isinstance(finish.get('delivered_final_runtime_reference_analysis'), dict) else {}
        rows.append(f"- delivered_final_runtime_reference: lufs={_brief_fmt(_deliv_ref.get('integrated_lufs'))}, source_role=`{(v60 or {}).get('v63162_source_role','n/a')}`")
    rows.append("")
    rows.append("## 11B. v64.0.3 Finishing Intelligence Selection Semantics Hotfix")
    if v6400:
        rows.append(f"- v64.0.3 finishing_intelligence: active={_brief_bool(v6400.get('active'))}, applied={_brief_bool(v6400.get('applied'))}, method=`{v6400.get('method','n/a')}`")
        rows.append(f"- playback_translation: phone={_brief_fmt(v6400_playback.get('phone_speaker_risk'))}, earbuds={_brief_fmt(v6400_playback.get('earbud_risk'))}, small_speaker={_brief_fmt(v6400_playback.get('small_speaker_risk'))}, car_low={_brief_fmt(v6400_playback.get('car_low_end_risk'))}, mono={_brief_fmt(v6400_playback.get('mono_translation_risk'))}, aggregate={_brief_fmt(v6400_playback.get('aggregate_translation_risk'))}")
        rows.append(f"- codec_stress: actual={_brief_bool(v6400_codec.get('actual_available'))}, risk={_brief_fmt(v6400_codec.get('risk'))}, worst_codec=`{v6400_codec.get('worst_codec') or 'n/a'}`, fallback=`{v6400_codec.get('fallback_reason') or '-'}`")
        _sel6400 = v6400_shadow.get('selected_candidate') if isinstance(v6400_shadow.get('selected_candidate'), dict) else {}
        rows.append(f"- shadow_selection: selected=`{v6400_shadow.get('selected_candidate_id','n/a')}`, delivery_accepted={_brief_bool(_sel6400.get('accepted'))}, delivery_rejected={_brief_bool(_sel6400.get('rejected'))}, quality_gate=`{_sel6400.get('quality_gate','n/a')}`, quality_rejected={_brief_bool(_sel6400.get('quality_rejected'))}, recommended_shadow=`{v6400_shadow.get('recommended_shadow_candidate_id','n/a')}`, audio_substitution={_brief_bool(v6400_shadow.get('selection_applied_to_audio'))}, reason=`{v6400_shadow.get('selected_reason','n/a')}`")
        _in_dmg = v6400_damage.get('input_damage') if isinstance(v6400_damage.get('input_damage'), dict) else {}
        _mst_dmg = v6400_damage.get('mastering_induced_damage') if isinstance(v6400_damage.get('mastering_induced_damage'), dict) else {}
        _cod_dmg = v6400_damage.get('codec_induced_damage') if isinstance(v6400_damage.get('codec_induced_damage'), dict) else {}
        _trn_dmg = v6400_damage.get('translation_induced_damage') if isinstance(v6400_damage.get('translation_induced_damage'), dict) else {}
        rows.append(f"- damage_ownership: input={_brief_bool(_in_dmg.get('owned'))}/{_brief_list(_in_dmg.get('flags'))}; mastering={_brief_bool(_mst_dmg.get('owned'))}/{_brief_list(_mst_dmg.get('flags'))}; codec={_brief_bool(_cod_dmg.get('owned'))}/{_brief_list(_cod_dmg.get('flags'))}; translation={_brief_bool(_trn_dmg.get('owned'))}/{_brief_list(_trn_dmg.get('flags'))}")
        rows.append(f"- proxy_actual_calibration: available={_brief_bool(v6400_cal.get('calibration_available'))}, delta={_brief_fmt(((v6400_cal.get('delta') or {}) if isinstance(v6400_cal.get('delta'), dict) else {}).get('codec_risk_delta_actual_minus_proxy'))}, underestimated={_brief_bool(v6400_cal.get('proxy_underestimated'))}")
        _auth6400 = v6400_replay.get('authority') if isinstance(v6400_replay.get('authority'), dict) else {}
        _dml6400 = _auth6400.get('dml_contract') if isinstance(_auth6400.get('dml_contract'), dict) else {}
        _tgt6400 = _auth6400.get('target_semantics') if isinstance(_auth6400.get('target_semantics'), dict) else {}
        rows.append(f"- decision_replay: path_steps={_brief_fmt(len(v6400_replay.get('decision_path') or []),0)}, dml_target_miss_allowed={_brief_bool(_dml6400.get('target_miss_allowed'))}, commercial_floor_met={_brief_bool(_tgt6400.get('commercial_floor_met'))}, target_reached={_brief_bool(_tgt6400.get('target_reached'))}")
        rows.append(f"- corpus_regression: case_id=`{v6400_reg.get('case_id','n/a')}`, status=`{v6400_reg.get('status','n/a')}`, pass_fail=`{v6400_reg.get('pass_fail','n/a')}`")
    else:
        rows.append("- v64.0.3 finishing_intelligence: not_available")
    rows.append("")
    rows.append("## 11C. v64.0.4 Commercial Maximum Density Push")
    if v6404:
        rows.append(f"- v64.0.4 density: active={_brief_bool(v6404.get('active'))}, applied={_brief_bool(v6404.get('applied'))}, mode=`{v6404.get('mode_policy','n/a')}`, selected=`{v6404_selected or 'n/a'}`")
        rows.append(f"- true_peak: target={_brief_fmt(v6404.get('target_true_peak_dbtp'))} dBTP, final={_brief_fmt(v6404.get('final_true_peak_dbtp'))} dBTP, honored={_brief_bool(v6404.get('true_peak_target_honored'))}, under_pushed={_brief_bool(v6404.get('under_pushed'))}")
        rows.append(f"- density_chain: push_gain={_brief_fmt(v6404.get('push_gain_db'))} dB, clipper_drive={_brief_fmt(v6404.get('clipper_drive_db'))} dB, limiter_gain={_brief_fmt(v6404.get('limiter_gain_db'))} dB, density={_brief_fmt(v6404.get('density_drive_amount'))}, parallel_mix={_brief_fmt(v6404.get('parallel_density_mix'))}")
        rows.append(f"- warnings: advisory={_brief_bool(v6404.get('warning_advisory'))}, blocking={_brief_bool(v6404.get('warning_blocking'))}, advisory_count={_brief_fmt(len(v6404_adv),0)}, blocking_reasons={_brief_list(v6404_blocks)}")
        _codec_sep = v6404.get('codec_safe_vs_maximum_wav_decision') if isinstance(v6404.get('codec_safe_vs_maximum_wav_decision'), dict) else {}
        rows.append(f"- codec_policy: selected=`{_codec_sep.get('selected','n/a')}`, codec_safe_candidate=`{_codec_sep.get('codec_safe_distribution_candidate','n/a')}`, maximum_wav=`{_codec_sep.get('maximum_wav_master_candidate','n/a')}`")
    else:
        rows.append("- v64.0.4 density: not_available")
    rows.append("")
    rows.append("## 11. Gemini Final Listener Review")
    rows.append(f"- available: {_brief_bool(gem.get('available'))}")
    rows.append(f"- verdict: `{gem.get('verdict', 'n/a')}`")
    rows.append(f"- commercial_readiness: {_brief_fmt(gem.get('commercial_readiness'))}")
    rows.append(f"- listener_impact: {_brief_fmt(gem.get('listener_impact'))}")
    rows.append(f"- improvement_targets: {_brief_list(gem.get('improvement_targets') or gem.get('risk_axis_watch'))}")
    rows.append("")
    rows.append("## 12. Quick Diagnosis")
    diagnosis = []
    if stage_m.get('attempted'):
        diagnosis.append('stage_m_stem_gate_attempted')
    if stage_m.get('selected_source') == 'stage_m_mixdown_candidate':
        diagnosis.append('stage_m_mixdown_selected')
    elif stage_m.get('selected_source') == 'stage_m_premaster_candidate':
        diagnosis.append('stage_m_legacy_premaster_selected_unexpected')
    elif stage_m:
        diagnosis.append('stage_m_original_stereo_route')
    _ft44_diag = report.get('v6314_4_commercial_floor_tolerance') if isinstance(report.get('v6314_4_commercial_floor_tolerance'), dict) else finish.get('v6314_4_commercial_floor_tolerance') if isinstance(finish.get('v6314_4_commercial_floor_tolerance'), dict) else {}
    if not finish.get('target_met'):
        diagnosis.append('commercial_floor_or_target_not_met')
    elif _ft44_diag.get('applied'):
        diagnosis.append('commercial_floor_met_within_v6314_4_tolerance')
    if v57_sel.get('delivered_final_has_damage'):
        diagnosis.append('delivered_final_damage_gate_active')
    if (v57.get('safe_push_budget_db') is not None) and str(v57.get('safe_push_budget_db')) in {'0', '0.0'}:
        diagnosis.append('safe_push_budget_zero')
    if v59_effective_active:
        diagnosis.append('v59_source_conditioning_triggered')
    if v59_2_effective_active:
        diagnosis.append('v59_2_program_dynamics_or_stereo_side_triggered')
    if v61.get('attempted'):
        diagnosis.append('v61_measured_feedback_optimizer_attempted')
    if v61.get('active'):
        diagnosis.append('v61_measured_feedback_optimizer_accepted')
    if v61.get('selected_pass') not in (None, 0, '0'):
        diagnosis.append('v61_feedback_selected_nonzero_pass')
    if v60.get('active'):
        diagnosis.append('v60_dml_shadow_or_controlled_candidate')
    if v60.get('applied'):
        diagnosis.append('v60_dml_applied')
    rows.append("- " + (", ".join(diagnosis) if diagnosis else "no_major_brief_flags"))
    rows.append("")
    rows.append("## 13. Files / Tables")
    rows.append("- This brief is the preferred handoff file for ChatGPT review. Use full report/stage_state only when this brief is insufficient.")
    rows.append("- Supabase structured tables remain the source of long-term learning/statistics; this file is a compact human/AI debug summary.")
    rows.append("")
    bamix = (report.get("busy_auto_mixing") if isinstance(report.get("busy_auto_mixing"), dict) else {})
    if not bamix and isinstance(state, dict):
        bamix = state.get("busy_auto_mixing") if isinstance(state.get("busy_auto_mixing"), dict) else {}
    if isinstance(bamix, dict) and bamix:
        rows.append("")
        rows.append("## 4A. Busy Auto Mixing")
        rows.append(f"- requested/enabled/available: {_brief_bool(bamix.get('requested'))} / {_brief_bool(bamix.get('enabled'))} / {_brief_bool(bamix.get('available'))}")
        rows.append(f"- status: `{bamix.get('status','n/a')}`, recipe: `{bamix.get('selected_mix_recipe','n/a')}`")
        rows.append(f"- premaster_used_for_mastering: {_brief_bool(bamix.get('premaster_used_for_mastering'))}, source=`{bamix.get('mastering_input_source','n/a')}`")
        rows.append(f"- candidate_render_count: {_brief_fmt(bamix.get('candidate_render_count'),0)}, correction_loops={_brief_fmt(bamix.get('correction_loops_executed'),0)}")
        _qc = bamix.get('premaster_qc') if isinstance(bamix.get('premaster_qc'), dict) else {}
        rows.append(f"- premaster_qc: LUFS={_brief_fmt(_qc.get('integrated_lufs'))}, TP={_brief_fmt(_qc.get('true_peak_dbtp'))}, crest={_brief_fmt(_qc.get('crest_factor_db'))}, LRA={_brief_fmt(_qc.get('lra_lu'))}, corr={_brief_fmt(_qc.get('phase_correlation'))}, warnings={_brief_list(_qc.get('warnings'))}")
        _hp = bamix.get('mastering_handoff_policy') if isinstance(bamix.get('mastering_handoff_policy'), dict) else {}
        if _hp:
            _gov = _hp.get('adaptive_damage_governor') if isinstance(_hp.get('adaptive_damage_governor'), dict) else {}
            _cr = _hp.get('commercial_readiness') if isinstance(_hp.get('commercial_readiness'), dict) else {}
            rows.append(f"- mastering_handoff: mode=`{_hp.get('handoff_mode') or 'n/a'}`, readiness={_brief_fmt(_cr.get('score'))}/{_cr.get('label','n/a')}, recipe=`{_hp.get('selected_mix_recipe') or 'n/a'}`, target={_brief_fmt(_hp.get('effective_final_target_lufs'))} LUFS, floor={_brief_fmt(_hp.get('effective_floor_lufs'))}, max_gain={_brief_fmt(_hp.get('max_total_gain_lufs'))} LU, finish_push={_brief_fmt(_hp.get('max_finish_push_db'))} dB, max_crest_loss={_brief_fmt(_hp.get('max_crest_loss_db'))} dB, governor={_brief_bool(_gov.get('active'))}, reasons={_brief_list(_gov.get('reasons'))}")
        _first_render = (bamix.get('render_attempts') or [None])[0] if isinstance(bamix.get('render_attempts'), list) and bamix.get('render_attempts') else {}
        _est = bamix.get('prerender_gain_estimator') if isinstance(bamix.get('prerender_gain_estimator'), dict) else {}
        if not _est and isinstance(_first_render, dict):
            _est = _first_render.get('prerender_gain_estimator') if isinstance(_first_render.get('prerender_gain_estimator'), dict) else {}
        if _est:
            _aligned = _est.get('aligned_proxy') if isinstance(_est.get('aligned_proxy'), dict) else {}
            _sel_win = _aligned.get('selected_window') if isinstance(_aligned.get('selected_window'), dict) else {}
            rows.append(f"- v63.1 estimator: initial_gain={_brief_fmt(_est.get('initial_premix_gain_db'))} dB, predicted_peak={_brief_fmt(_est.get('predicted_uncorrected_sum_peak_dbfs'))} dBFS, aligned_peak={_brief_fmt(_sel_win.get('peak_dbfs'))} dBFS, safety={_brief_fmt(_est.get('calibrated_proxy_upper_safety_db'))} dB, authority=`{_est.get('authority','n/a')}`")
        _efb = bamix.get('v6313_2_prerender_estimator_feedback') if isinstance(bamix.get('v6313_2_prerender_estimator_feedback'), dict) else (bamix.get('v6313_prerender_estimator_feedback') if isinstance(bamix.get('v6313_prerender_estimator_feedback'), dict) else {})
        if _efb:
            rows.append(f"- v63.1 estimator_feedback: first_underdriven={_brief_bool(_efb.get('first_render_underdriven'))}, soft_correction={_brief_bool(_efb.get('soft_target_correction_applied'))}, correction_gain={_brief_fmt(_efb.get('correction_gain_db'))} dB, recommendation=`{_efb.get('recommendation','n/a')}`")
        _v631 = bamix.get('v631_mix_strategy') if isinstance(bamix.get('v631_mix_strategy'), dict) else {}
        if _v631:
            _mods = _v631.get('module_intensity') if isinstance(_v631.get('module_intensity'), dict) else {}
            rows.append(f"- v63.1 modules: active={_brief_bool(_v631.get('active'))}, glue={_mods.get('glue')}, vocal_pocket={_mods.get('vocal_pocket')}, kick_bass={_mods.get('kick_bass')}, drum_punch={_mods.get('drum_punch')}, harmonic_density={_mods.get('harmonic_density')}, elliptical={_mods.get('elliptical')}")
        _render = (bamix.get('render_attempts') or [])[-1] if isinstance(bamix.get('render_attempts'), list) and bamix.get('render_attempts') else {}
        _modrun = ((_render.get('v631_professional_module_render') or {}).get('module_runtime') if isinstance(_render.get('v631_professional_module_render'), dict) else {})
        if isinstance(_modrun, dict) and _modrun:
            _glue = _modrun.get('glue') if isinstance(_modrun.get('glue'), dict) else {}
            _kb = _modrun.get('kick_bass') if isinstance(_modrun.get('kick_bass'), dict) else {}
            _vp = _modrun.get('vocal_pocket') if isinstance(_modrun.get('vocal_pocket'), dict) else {}
            _kb_duck_avg = _kb.get('avg_duck_db', _kb.get('last_duck_db'))
            _kb_duck_max = _kb.get('max_duck_db', _kb.get('last_duck_db'))
            _vp_cut_avg = _vp.get('avg_pocket_cut_db', _vp.get('last_cut_db'))
            _vp_cut_max = _vp.get('max_pocket_cut_db', _vp.get('last_cut_db'))
            _glue_avg_raw = _glue.get('avg_gr_abs_db', _glue.get('avg_glue_gr_db', _glue.get('avg_gr_db')))
            _glue_max_raw = _glue.get('max_gr_abs_db', _glue.get('max_glue_gr_db', _glue.get('max_gr_db')))
            try:
                _glue_avg = abs(float(_glue_avg_raw))
            except Exception:
                _glue_avg = _glue_avg_raw
            try:
                _glue_max = abs(float(_glue_max_raw))
            except Exception:
                _glue_max = _glue_max_raw
            rows.append(f"- v63.1 runtime: blocks={_brief_fmt(_modrun.get('blocks_processed'),0)}, glue_gr_abs_avg/max={_brief_fmt(_glue_avg)}/{_brief_fmt(_glue_max)}, kick_bass_duck_avg/max={_brief_fmt(_kb_duck_avg)}/{_brief_fmt(_kb_duck_max)}, vocal_pocket_cut_avg/max={_brief_fmt(_vp_cut_avg)}/{_brief_fmt(_vp_cut_max)}")
            _aug = _modrun.get('stem_augmentation') if isinstance(_modrun.get('stem_augmentation'), dict) else {}
            if _aug:
                def _aug_one(_name):
                    _m = _aug.get(_name) if isinstance(_aug.get(_name), dict) else {}
                    if not _m:
                        return f"{_name}=n/a"
                    _applied = _brief_bool(_m.get('applied'))
                    _frac = _brief_fmt(_m.get('applied_fraction'))
                    _extra = []
                    for _k in ('duck_db', 'blend_db', 'side_high_over_mid_high_db', 'need_score', 'scale_min'):
                        if _m.get(_k) is not None:
                            _extra.append(f"{_k}={_brief_fmt(_m.get(_k))}")
                    return f"{_name}={_applied}/{_frac}" + (" (" + ', '.join(_extra[:3]) + ")" if _extra else "")
                rows.append("- v63.3 stem_augmentation: " + "; ".join([
                    _aug_one('bass_harmonic_translation'),
                    _aug_one('low_mid_body_fill'),
                    _aug_one('vocal_support_body_layer'),
                    _aug_one('center_anchor'),
                    _aug_one('drum_parallel_density'),
                    _aug_one('transient_ghost'),
                    _aug_one('side_texture_control'),
                    _aug_one('bus_peak_guard'),
                ]))
                _v6370 = bamix.get('v6370_body_center_vocal_support') if isinstance(bamix.get('v6370_body_center_vocal_support'), dict) else {}
                if _v6370:
                    _mods6370 = _v6370.get('modules') if isinstance(_v6370.get('modules'), dict) else {}
                    _lm = _mods6370.get('low_mid_body_fill') if isinstance(_mods6370.get('low_mid_body_fill'), dict) else {}
                    _vs = _mods6370.get('vocal_support_body_layer') if isinstance(_mods6370.get('vocal_support_body_layer'), dict) else {}
                    _ca = _mods6370.get('center_anchor') if isinstance(_mods6370.get('center_anchor'), dict) else {}
                    _guards = _v6370.get('guards') if isinstance(_v6370.get('guards'), dict) else {}
                    rows.append(f"- v63.7 body_center_vocal_support: active={_brief_bool(_v6370.get('active'))}, applied={_brief_bool(_v6370.get('applied'))}, low_mid={_brief_bool(_lm.get('applied'))}/{_brief_fmt(_lm.get('assist_rms_db'))} dB, vocal_body={_brief_bool(_vs.get('applied'))}/{_brief_fmt(_vs.get('assist_rms_db'))} dB, center={_brief_bool(_ca.get('applied'))}/{_brief_fmt(_ca.get('anchor_rms_db'))} dB, mud_rb={_brief_fmt(_lm.get('mud_rollback_db'))}/{_brief_fmt(_vs.get('mud_rollback_db'))} dB, side_lm_trim={_brief_fmt(_ca.get('side_lowmid_trim_db'))} dB, bus_guard_scale_min={_brief_fmt(_guards.get('bus_peak_guard_scale_min'))}")
                _v6380 = bamix.get('v6380_bass_harmonic_translation') if isinstance(bamix.get('v6380_bass_harmonic_translation'), dict) else (bamix.get('bass_harmonic_translation') if isinstance(bamix.get('bass_harmonic_translation'), dict) else {})
                if _v6380:
                    _hg6380 = _v6380.get('harmonic_generation') if isinstance(_v6380.get('harmonic_generation'), dict) else {}
                    _ma6380 = _v6380.get('mono_anchor') if isinstance(_v6380.get('mono_anchor'), dict) else {}
                    _dc6380 = _v6380.get('dc_blocker') if isinstance(_v6380.get('dc_blocker'), dict) else {}
                    _vc6380 = _v6380.get('vocal_low_mid_conflict_notch') if isinstance(_v6380.get('vocal_low_mid_conflict_notch'), dict) else {}
                    _gd6380 = _v6380.get('guards') if isinstance(_v6380.get('guards'), dict) else {}
                    rows.append(f"- v63.8 bass_harmonic_translation: active={_brief_bool(_v6380.get('active'))}, applied={_brief_bool(_v6380.get('applied'))}, amount={_brief_fmt(_v6380.get('bass_translation_amount'))}, h2/h3={_brief_fmt(_v6380.get('second_harmonic_ratio', _hg6380.get('second_harmonic_ratio')))}/{_brief_fmt(_v6380.get('third_harmonic_ratio', _hg6380.get('third_harmonic_ratio')))}, mono_anchor={_brief_bool(_ma6380.get('active'))}/side_trim={_brief_fmt(_ma6380.get('side_low_trim_db'))} dB, dc_blocker={_brief_bool(_dc6380.get('active'))}/{_brief_fmt(_dc6380.get('hp_hz'))} Hz, vocal_notch={_brief_bool(_vc6380.get('applied'))}/{_brief_fmt(_vc6380.get('duck_db'))} dB, bus_guard_scale_min={_brief_fmt(_gd6380.get('bus_peak_guard_scale_min'))}")
                _v6390 = bamix.get('v6390_side_texture_stereo_cleanliness') if isinstance(bamix.get('v6390_side_texture_stereo_cleanliness'), dict) else (bamix.get('side_texture_stereo_cleanliness') if isinstance(bamix.get('side_texture_stereo_cleanliness'), dict) else {})
                if _v6390:
                    _stc6390 = _v6390.get('side_texture_control') if isinstance(_v6390.get('side_texture_control'), dict) else {}
                    _hf6390 = _v6390.get('side_high_hash_fizz_suppressor') if isinstance(_v6390.get('side_high_hash_fizz_suppressor'), dict) else {}
                    _sg6390 = _v6390.get('stereo_cleanliness_guard') if isinstance(_v6390.get('stereo_cleanliness_guard'), dict) else {}
                    _amb6390 = _v6390.get('ambience_collapse_detection') if isinstance(_v6390.get('ambience_collapse_detection'), dict) else {}
                    _mono6390 = _v6390.get('mono_fold_down_rollback') if isinstance(_v6390.get('mono_fold_down_rollback'), dict) else {}
                    rows.append(f"- v63.9 side_texture_stereo_cleanliness: active={_brief_bool(_v6390.get('active'))}, applied={_brief_bool(_v6390.get('applied'))}, side_control={_brief_bool(_stc6390.get('applied'))}, hash_risk={_brief_fmt(_hf6390.get('hash_risk'))}, suppression={_brief_fmt(_hf6390.get('suppression_db'))} dB, band={_hf6390.get('detected_band_hz') or _stc6390.get('affected_band_hz')}, ambience_risk={_brief_fmt(_amb6390.get('risk'))}/rollback={_brief_bool(_amb6390.get('rollback_active'))}, mono_risk={_brief_fmt(_mono6390.get('compatibility_risk'))}/rollback={_brief_bool(_mono6390.get('rollback_active'))}, center_protect={_brief_bool(_sg6390.get('center_protection_active'))}")
            # v63.1.4.2.1: target-guard visibility hotfix.  The standard render
            # writes processing_report at the report top level, while some older
            # two-stage paths mirror it under pre_limiter_report.processing_report.
            # Check both locations and print the marker even when inactive so the
            # next debug brief can prove whether the guard actually had authority.
            _top_pr = report.get('processing_report') if isinstance(report.get('processing_report'), dict) else {}
            _pre_pr = ((report.get('pre_limiter_report') or {}).get('processing_report') if isinstance(report.get('pre_limiter_report'), dict) else {})
            _tg = {}
            for _pr in (_top_pr, _pre_pr):
                if not isinstance(_pr, dict):
                    continue
                _cand = _pr.get('v6314_2_bamix_loudness_target_guard')
                if not isinstance(_cand, dict):
                    _cand = (_pr.get('loudness_iteration') or {}).get('v6314_2_bamix_loudness_target_guard') if isinstance(_pr.get('loudness_iteration'), dict) else {}
                if isinstance(_cand, dict) and _cand:
                    _tg = _cand
                    break
            if isinstance(_tg, dict) and _tg:
                rows.append(f"- v63.1.4.2 target_guard: active={_brief_bool(_tg.get('active'))}, applied={_brief_bool(_tg.get('applied'))}, original={_brief_fmt(_tg.get('original_target_lufs'))}, effective={_brief_fmt(_tg.get('effective_target_lufs'))}, floor={_brief_fmt(_tg.get('effective_floor_lufs'))}, reduction={_brief_fmt(_tg.get('target_reduction_db'))} dB, reason=`{_tg.get('reason') or 'n/a'}`")
            else:
                rows.append("- v63.1.4.2 target_guard: marker=`missing` reason=`audio_engine_mastering_target_guard_not_present_or_not_used`")
            _auth = bamix.get('v6313_2_module_authority_summary') if isinstance(bamix.get('v6313_2_module_authority_summary'), dict) else {}
            if _auth:
                _ag = _auth.get('glue') if isinstance(_auth.get('glue'), dict) else {}
                _ak = _auth.get('kick_bass') if isinstance(_auth.get('kick_bass'), dict) else {}
                _av = _auth.get('vocal_pocket') if isinstance(_auth.get('vocal_pocket'), dict) else {}
                rows.append(f"- v63.1 authority: glue_need_avg/max={_brief_fmt(_ag.get('need_score_avg'))}/{_brief_fmt(_ag.get('need_score_max'))}, glue_label=`{_ag.get('module_authority_decision_label') or _ag.get('need_actual_alignment') or _ag.get('aggregate_decision') or 'n/a'}`, kick_label=`{_ak.get('module_authority_decision_label') or _ak.get('need_actual_alignment') or 'n/a'}`, vocal_label=`{_av.get('module_authority_decision_label') or _av.get('need_actual_alignment') or 'n/a'}`)")

    return "\n".join(rows)


def _write_debug_brief_if_requested(args: argparse.Namespace, report: dict[str, Any], state: dict[str, Any] | None, log_fp) -> dict[str, Any]:
    if not _env_on("BUSY_DEBUG_BRIEF_MD", "1"):
        return {"enabled": False, "active": False, "reason": "disabled"}
    try:
        local = Path(args.report).resolve().with_name("busy_debug_brief.md")
        local.write_text(_build_debug_brief_markdown(report if isinstance(report, dict) else {}, state if isinstance(state, dict) else {}, args), encoding="utf-8")
        supa = _debug_brief_supabase_path(args)
        uploaded = False
        if supa and _env_on("BUSY_DEBUG_BRIEF_UPLOAD", "1"):
            try:
                storage = SupabaseStorage.from_env()
                storage.upload_file(supa, str(local), content_type="text/markdown; charset=utf-8")
                uploaded = True
            except Exception as exc:
                _log(log_fp, "debug_brief_upload_failed", error=str(exc)[:300], path=supa)
        meta = {"schema_version": "busy_debug_brief_v8_5_3_64_0_3_finishing_intelligence_selection_semantics_hotfix", "enabled": True, "active": True, "local_path": str(local), "supabase_path": supa, "uploaded": uploaded, "bytes": local.stat().st_size}
        _log(log_fp, "debug_brief_written", **meta)
        return meta
    except Exception as exc:
        _log(log_fp, "debug_brief_failed", error_type=type(exc).__name__, error=str(exc)[:500])
        return {"schema_version": "busy_debug_brief_v8_5_3_64_0_3_finishing_intelligence_selection_semantics_hotfix", "enabled": True, "active": False, "reason": "error", "error_type": type(exc).__name__, "error": str(exc)[:300]}

def _upload_stage_state_if_requested(args: argparse.Namespace, log_fp, storage: SupabaseStorage | None = None) -> None:
    if not getattr(args, "supabase_stage_state", None):
        return
    if not getattr(args, "stage_state", None):
        raise RuntimeError("--stage-state is required when --supabase-stage-state is used")
    local_state = Path(args.stage_state)
    if not local_state.exists():
        raise RuntimeError(f"Stage state file does not exist: {local_state}")
    _log(log_fp, "stage_start_stage_state_upload", path=args.supabase_stage_state)
    storage = storage or SupabaseStorage.from_env()
    storage.upload_file(args.supabase_stage_state, str(local_state), content_type="application/json")
    _log(log_fp, "stage_done_stage_state_upload", path=args.supabase_stage_state, bytes=local_state.stat().st_size)


def _download_stage_state_if_requested(args: argparse.Namespace, log_fp, storage: SupabaseStorage | None = None) -> None:
    if not getattr(args, "supabase_stage_state", None):
        return
    if not getattr(args, "stage_state", None):
        raise RuntimeError("--stage-state is required when --supabase-stage-state is used")
    local_state = Path(args.stage_state)
    local_state.parent.mkdir(parents=True, exist_ok=True)
    _log(log_fp, "stage_start_stage_state_download", path=args.supabase_stage_state)
    storage = storage or SupabaseStorage.from_env()
    storage.download_file(args.supabase_stage_state, str(local_state))
    _log(log_fp, "stage_done_stage_state_download", local=str(local_state), bytes=local_state.stat().st_size)

def _two_stage_enabled() -> bool:
    return str(os.environ.get("BUSY_TWO_STAGE_LIMITING", "1")).strip().lower() in {"1", "true", "yes", "on", "y"}


def _intermediate_path_from_output(output_path: str) -> str:
    clean = str(output_path).strip().lstrip("/")
    if "/output/" in clean:
        return clean.replace("/output/output_48k_24bit.wav", "/intermediate/pre_limiter_float32.wav").replace("/output/", "/intermediate/")
    return clean.rstrip("/") + ".pre_limiter_float32.wav"


def _tail_text_file(path: str | Path, max_chars: int = 6000) -> str:
    try:
        p = Path(path)
        if not p.exists():
            return ""
        txt = p.read_text(encoding="utf-8", errors="replace")
        return txt[-max_chars:]
    except Exception as exc:
        return f"<tail_read_failed: {type(exc).__name__}: {exc}>"


def _child_log_path_from_cmd(cmd: list[str]) -> str | None:
    try:
        if "--log" in cmd:
            i = cmd.index("--log")
            if i + 1 < len(cmd):
                return cmd[i + 1]
    except Exception:
        pass
    return None


def _run_child(cmd: list[str], log_fp, stage_name: str, timeout_sec: int = 1800, env: dict[str, str] | None = None) -> int:
    _log(log_fp, f"stage_start_{stage_name}", cmd=" ".join(cmd))
    child_log = _child_log_path_from_cmd(cmd)
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env or os.environ.copy(),
        cwd=str(Path(__file__).resolve().parent),
    )
    t0 = time.time()
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        # Re-emit child BUSY_JOB lines so the Streamlit progress parser sees them.
        print(line, flush=True)
        if log_fp:
            log_fp.write(line + "\n")
            log_fp.flush()
        if time.time() - t0 > timeout_sec:
            try:
                proc.kill()
            except Exception:
                pass
            tail = _tail_text_file(child_log) if child_log else ""
            _log(log_fp, f"stage_timeout_{stage_name}", timeout_sec=timeout_sec, child_log=child_log, child_log_tail=tail)
            return 124
    rc_raw = int(proc.wait() or 0)
    rc = rc_raw
    signal_num = None
    # v62.9.6.6.2: Python subprocess returns negative codes when a child
    # is terminated by a signal.  Returning that negative value through
    # sys.exit() wraps it to 256-N (for SIGKILL, -9 becomes 247), which made
    # Cloud Run reports read as the confusing "return code 247".  Normalize
    # to the conventional 128+signal code and log the real signal.
    if rc_raw < 0:
        signal_num = abs(rc_raw)
        rc = 128 + signal_num
    elapsed = round(time.time() - t0, 3)
    if rc != 0:
        tail = _tail_text_file(child_log) if child_log else ""
        _log(log_fp, f"stage_failed_{stage_name}", returncode=rc, raw_returncode=rc_raw, terminated_by_signal=signal_num, elapsed_sec=elapsed, child_log=child_log, child_log_tail=tail)
    _log(log_fp, f"stage_done_{stage_name}", returncode=rc, raw_returncode=rc_raw, terminated_by_signal=signal_num, elapsed_sec=elapsed)
    return rc

def _build_features_and_decision(local_input: Path, private_zip: str, mode: str, log_fp, user_style_hint: str = "", private_rulepack_path: str = "", stem_zip_path: Path | None = None, work_dir: Path | None = None) -> tuple[np.ndarray, int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    _log(log_fp, "stage_start_load_audio")
    analysis_dtype = os.environ.get("BUSY_ANALYSIS_AUDIO_DTYPE", "float32").strip() or "float32"
    y, sr = load_audio_file(str(local_input), target_sr=48_000, dtype=analysis_dtype)
    _log(log_fp, "stage_done_load_audio", sr=sr, shape=list(y.shape), dtype=str(y.dtype), seconds=round(y.shape[0] / sr, 3), analysis_dtype=analysis_dtype)

    _log(log_fp, "stage_start_preflight_clean")
    y, preflight_report = preflight_clean_audio(y, sr)
    _log(log_fp, "stage_done_preflight_clean", **preflight_report)

    parallel_analysis = str(os.environ.get("BUSY_PARALLEL_ANALYSIS", "1")).lower() in {"1", "true", "yes", "on"}
    full_duration_sec = float(y.shape[0] / max(sr, 1)) if getattr(y, "shape", None) is not None else 0.0
    analysis_max_sec = float(os.environ.get("BUSY_ANALYZE_AUDIO_MAX_SEC", "120") or "120")
    analysis_samples = int(max(30.0, analysis_max_sec) * sr)
    y_analysis = y[:analysis_samples] if full_duration_sec > analysis_max_sec and y.shape[0] > analysis_samples else y
    if y_analysis is not y:
        _log(log_fp, "stage_info_analysis_subset", full_seconds=round(full_duration_sec, 3), analysis_seconds=round(y_analysis.shape[0] / sr, 3), reason="memory_guard")

    if parallel_analysis:
        _log(log_fp, "stage_start_parallel_analysis")
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_features = ex.submit(analyze_audio, y_analysis, sr)
            fut_segments = ex.submit(analyze_segments, y, sr, window_sec=10.0, max_segments=24)
            features = fut_features.result()
            segment_analysis = fut_segments.result()
        features["segment_analysis"] = segment_analysis
        features["preflight_clean"] = preflight_report
        _log(log_fp, "stage_done_parallel_analysis", lufs=features.get("integrated_lufs"), true_peak=features.get("approx_true_peak_dbfs"), crest=features.get("crest_factor_db"), segments=segment_analysis.get("summary", {}).get("segments_analyzed") or len(segment_analysis.get("segments", [])), loudest=segment_analysis.get("summary", {}).get("loudest_10s", {}).get("start_sec"), brightest=segment_analysis.get("summary", {}).get("brightest_10s", {}).get("start_sec"))
    else:
        _log(log_fp, "stage_start_analyze_audio")
        features = analyze_audio(y_analysis, sr)
        if y_analysis is not y:
            features["duration_sec"] = round(full_duration_sec, 3)
            features["analysis_window_sec"] = round(float(y_analysis.shape[0] / sr), 3)
        features["preflight_clean"] = preflight_report
        _log(log_fp, "stage_done_analyze_audio", lufs=features.get("integrated_lufs"), true_peak=features.get("approx_true_peak_dbfs"), crest=features.get("crest_factor_db"))

        _log(log_fp, "stage_start_segment_analysis")
        segment_analysis = analyze_segments(y, sr, window_sec=10.0, max_segments=24)
        features["segment_analysis"] = segment_analysis
        _log(log_fp, "stage_done_segment_analysis", segments=segment_analysis.get("summary", {}).get("segments_analyzed") or len(segment_analysis.get("segments", [])), loudest=segment_analysis.get("summary", {}).get("loudest_10s", {}).get("start_sec"), brightest=segment_analysis.get("summary", {}).get("brightest_10s", {}).get("start_sec"))

    # v8.5.3.34: optimized architecture metadata.  Pre-clean is represented
    # as a JSON/ChainSpec plan and in-memory validation basis, not an exported
    # intermediate WAV.  preflight_clean_audio has already applied the minimal
    # technical cleanup in-memory.
    try:
        pre_clean_plan = build_pre_clean_plan(features, preflight_report)
        post_pre_clean_analysis = build_post_pre_clean_analysis(features, pre_clean_plan)
        features["pre_clean_plan"] = pre_clean_plan
        features["post_pre_clean_analysis"] = post_pre_clean_analysis
        features["optimized_pipeline_architecture"] = build_pipeline_architecture_state(features, {}, final_intensity=None)
        _log(
            log_fp,
            "stage_done_pre_clean_plan",
            actions=len(pre_clean_plan.get("actions") or []),
            intermediate_wav_written=False,
            policy=pre_clean_plan.get("persistence"),
        )
    except Exception as exc:
        features["pre_clean_plan"] = {"enabled": False, "reason": "exception", "error": str(exc)[:500]}
        features["post_pre_clean_analysis"] = {"enabled": False, "reason": "exception", "error": str(exc)[:500]}
        _log(log_fp, "stage_error_pre_clean_plan", error=str(exc)[:500])

    _log(log_fp, "stage_start_essentia_evidence")
    essentia_evidence = extract_essentia_evidence(y_analysis, sr)
    features["essentia_evidence"] = essentia_evidence
    _log(
        log_fp,
        "stage_done_essentia_evidence",
        available=essentia_evidence.get("available"),
        reason=essentia_evidence.get("reason"),
        seconds_analyzed=essentia_evidence.get("seconds_analyzed"),
        guitar_texture_score=(essentia_evidence.get("evidence_scores", {}) or {}).get("guitar_texture_score") if isinstance(essentia_evidence.get("evidence_scores"), dict) else None,
        electronic_texture_score=(essentia_evidence.get("evidence_scores", {}) or {}).get("electronic_texture_score") if isinstance(essentia_evidence.get("evidence_scores"), dict) else None,
    )

    # v8.5.3.62.9.7: Stem Evidence Assist runs before Suno/AI artifact
    # analysis and GPT decision.  It is analysis-only/control-map evidence;
    # Stage M mixdown/rendering is replaced and separated audio is never injected.
    try:
        if stem_zip_path is not None and Path(stem_zip_path).exists() and _env_on("BUSY_STEM_EVIDENCE_ASSIST", "1"):
            _log(log_fp, "stage_start_stem_evidence_assist_pre_artifact", zip=str(stem_zip_path), audio_injection=False)
            stem_bundle = analyze_stem_evidence_assist(local_input, stem_zip_path, work_dir=work_dir or Path(local_input).parent)
            assist_pre = stem_bundle.get("multi_component_assist") if isinstance(stem_bundle.get("multi_component_assist"), dict) else {}
            stem_evidence = stem_bundle.get("stem_evidence_assist") if isinstance(stem_bundle.get("stem_evidence_assist"), dict) else build_stem_evidence_pack(assist_pre)
            features["stem_evidence_assist"] = stem_evidence
            features["multi_component_assist"] = assist_pre
            features["low_grs_clean_component_assist"] = assist_pre.get("low_grs_clean_component_assist", {}) if isinstance(assist_pre, dict) else {}
            features["stem_sum_residue_witness"] = assist_pre.get("stem_sum_residue_witness", {}) if isinstance(assist_pre, dict) else {}
            features["multi_component_assist_summary"] = {
                "provided": assist_pre.get("provided"),
                "usable": assist_pre.get("usable"),
                "recommended_mode": assist_pre.get("recommended_mode"),
                "track_count_analyzed": assist_pre.get("track_count_analyzed"),
                "stage_m_mixdown_builder_present": False,
                "stage_m_mixdown_accepted": False,
                "stage_m_removed": True,
                "stem_evidence_assist_global_reliability": stem_evidence.get("global_reliability"),
                "stem_audio_injection_forbidden": True,
                "separated_tracks_used_for_audio_render": False,
            }
            _log(
                log_fp,
                "stage_done_stem_evidence_assist_pre_artifact",
                provided=stem_evidence.get("provided"),
                usable=stem_evidence.get("usable"),
                global_reliability=stem_evidence.get("global_reliability"),
                roles=list((stem_evidence.get("role_map") or {}).keys())[:12] if isinstance(stem_evidence.get("role_map"), dict) else [],
                side_high_hash=((stem_evidence.get("residue_witness") or {}).get("side_high_hash_risk") if isinstance(stem_evidence.get("residue_witness"), dict) else None),
                audio_injection=False,
            )
        else:
            features["stem_evidence_assist"] = {
                "schema_version": "busy_stem_evidence_assist_v8_5_3_62_9_7",
                "enabled": _env_on("BUSY_STEM_EVIDENCE_ASSIST", "1"),
                "provided": bool(stem_zip_path),
                "usable": False,
                "reason": "stem_zip_missing_or_disabled",
                "audio_injection": False,
                "audio_injection_allowed": False,
                "stage_m_mixdown_replaced": True,
            }
    except Exception as exc:
        features["stem_evidence_assist"] = {
            "schema_version": "busy_stem_evidence_assist_v8_5_3_62_9_7",
            "enabled": True,
            "provided": bool(stem_zip_path),
            "usable": False,
            "reason": "exception",
            "error": str(exc)[:500],
            "audio_injection": False,
            "audio_injection_allowed": False,
            "stage_m_mixdown_replaced": True,
        }
        _log(log_fp, "stage_error_stem_evidence_assist_pre_artifact", error=str(exc)[:500])

    _log(log_fp, "stage_start_suno_artifact_analysis")
    suno_artifact_analysis = analyze_suno_artifacts(features, features.get("segment_analysis", {}))
    features["suno_artifact_analysis"] = suno_artifact_analysis
    _log(log_fp, "stage_done_suno_artifact_analysis", overall_risk=suno_artifact_analysis.get("overall_risk"), repair_recommended=suno_artifact_analysis.get("repair_recommended"), active_repairs=suno_artifact_analysis.get("active_repairs"))

    clean_hint = _clean_user_style_hint(user_style_hint or os.environ.get("BUSY_USER_STYLE_HINT"))
    if clean_hint:
        features["user_style_hint"] = {
            "provided": True,
            "raw_text": clean_hint,
            "source": "user_optional_style_tag",
            "trust_policy": "weighted_hint_cross_checked_against_audio_features",
        }
        _log(log_fp, "stage_done_user_style_hint", provided=True, chars=len(clean_hint))
    else:
        features["user_style_hint"] = {"provided": False}

    _log(log_fp, "stage_start_private_rules")
    zip_password = os.environ.get("PRIVATE_DOCS_ZIP_PASSWORD")
    private_bundle = load_private_bundle(private_zip, zip_password)
    private_rules = private_bundle.get("markdown", "")
    private_json = private_bundle.get("json", {}) or {}
    private_profile_db = None
    private_profile_asset = None
    # Prefer the newest private reference DB shipped in private-assets/private_docs_encrypted.zip.
    # v9/v9.1 contains the broadest exact_profiles, family_profiles and genre_alias_map;
    # v7 remains a compatibility fallback.
    preferred_markers = [
        "genre_mastering_reference_v9",
        "genre_mastering_reference_v7",
        "genre_mastering_reference_v6",
        "genre_mastering_reference",
    ]
    for marker in preferred_markers:
        for asset_name, asset in sorted(private_json.items()):
            if marker in asset_name and str(asset_name).endswith(".json"):
                private_profile_db = asset
                private_profile_asset = asset_name
                break
        if private_profile_db is not None:
            break
    private_genre_spectral_reference_db = None
    private_genre_spectral_reference_asset = None
    # v8.5.3.9: detailed genre/profile stack uses genre spectral/loudness
    # reference data as a private replaceable asset.  It is not a secret key, but
    # it should live with private reference DB assets rather than being baked into
    # the code patch so it can be updated independently.
    spectral_markers = [
        "genre_spectral_reference_v8",
        "genre_spectral_loudness_reference",
        "genre_spectral_reference",
        "spectral_loudness_reference",
    ]
    for marker in spectral_markers:
        for asset_name, asset in sorted(private_json.items()):
            low_name = str(asset_name).lower()
            if marker in low_name and low_name.endswith(".json") and isinstance(asset, dict) and isinstance(asset.get("profiles"), dict):
                private_genre_spectral_reference_db = asset
                private_genre_spectral_reference_asset = asset_name
                break
        if private_genre_spectral_reference_db is not None:
            break
    if private_genre_spectral_reference_db is not None:
        features["private_genre_spectral_reference_db"] = private_genre_spectral_reference_db
        features.setdefault("private_asset_manifest", {})["genre_spectral_reference_asset"] = private_genre_spectral_reference_asset
    private_profile_is_v7 = is_v7_db(private_profile_db)
    private_rulepack = load_private_rulepack(private_rulepack_path or os.environ.get("PRIVATE_RULEPACK_RESOLVED_PATH", ""), os.environ.get("PRIVATE_RULEPACK_ZIP_PASSWORD") or zip_password)
    features["private_rulepack"] = private_rulepack if private_rulepack.get("enabled") else {"enabled": False, "reason": private_rulepack.get("reason") or "not_loaded"}
    _log(log_fp, "stage_done_private_rules", markdown_chars=len(private_rules), has_profile_db=private_profile_db is not None, profile_asset=private_profile_asset, private_profile_is_v7=private_profile_is_v7, has_genre_spectral_reference=private_genre_spectral_reference_db is not None, genre_spectral_reference_asset=private_genre_spectral_reference_asset, private_rulepack_enabled=bool(private_rulepack.get("enabled")), private_rulepack_file_count=private_rulepack.get("raw_file_count"))

    # v8.5.0: compute Essentia/Rule/private capability evidence before audio element extractors.
    # Gemini/Realtime mini receive this compact evidence pack together with the
    # same element-first preview, so they are not blind broad-genre guessers. They must
    # confirm or reject the evidence by listening.
    try:
        _log(log_fp, "stage_start_pre_audio_genre_evidence")
        pre_audio_ge = build_genre_evidence(features, private_profile_db or {})
        features["pre_audio_genre_evidence"] = pre_audio_ge
        features["evidence_informed_audio_judge_context"] = _compact_pre_audio_evidence(features, pre_audio_ge)
        _log(
            log_fp,
            "stage_done_pre_audio_genre_evidence",
            top_profile=pre_audio_ge.get("top_profile"),
            top_db_profile=pre_audio_ge.get("top_db_profile"),
            role_count=len(((pre_audio_ge.get("hybrid_role_evidence") or {}).get("roles") or {}) if isinstance(pre_audio_ge.get("hybrid_role_evidence"), dict) else {}),
            has_private_capability=bool((pre_audio_ge.get("private_capability_expansion") or {}).get("enabled") if isinstance(pre_audio_ge.get("private_capability_expansion"), dict) else False),
        )
    except Exception as exc:
        features["pre_audio_genre_evidence"] = {"available": False, "reason": "pre_audio_genre_evidence_exception", "error": str(exc)[:500]}
        features["evidence_informed_audio_judge_context"] = {"available": False, "reason": "pre_audio_genre_evidence_exception", "error": str(exc)[:500]}
        _log(log_fp, "stage_error_pre_audio_genre_evidence", error=str(exc)[:500])

    # v8.5.3.48.2.1: AI resource role separation.
    # Realtime remains the default early listener/comparator.  Gemini is no
    # longer spent on the same raw-audio preview just because
    # BUSY_GEMINI_AUDIO_JUDGE=1 is set; in role-separated mode it is reserved for
    # validation/final-quality checks.  Restore the old duplicate raw Gemini pass
    # only with BUSY_GEMINI_RAW_AUDIO_JUDGE=1 or BUSY_AI_ROLE_SEPARATION=0.
    ai_role_separation_enabled = _env_on("BUSY_AI_ROLE_SEPARATION", "1")
    if ai_role_separation_enabled:
        gemini_enabled = _env_on("BUSY_GEMINI_RAW_AUDIO_JUDGE", "0")
    else:
        gemini_enabled = _env_on("BUSY_GEMINI_AUDIO_JUDGE", os.environ.get("BUSY_GEMINI_RAW_AUDIO_JUDGE", "0"))
    openai_audio_enabled = _env_on("BUSY_OPENAI_AUDIO_JUDGE", os.environ.get("BUSY_REALTIME_GENRE_JUDGE", "1"))
    features["ai_resource_allocation_strategy"] = {
        "schema_version": "busy_ai_resource_allocation_v8_5_3_55_3",
        "role_separation_enabled": bool(ai_role_separation_enabled),
        "realtime_role": "early_raw_audio_listener_and_genre_element_judge",
        "gemini_role": "post_export_final_master_highlight_listener_review_not_raw_genre_judge",
        "gpt54mini_role": "text_decision_summary_public_safe_explanation_packet_builder",
        "gpt55_role": "hard_case_arbitration_within_allowed_candidates",
        "raw_gemini_enabled": bool(gemini_enabled),
        "raw_realtime_enabled": bool(openai_audio_enabled),
        "hard_qc_is_law": True,
        "no_ai_dsp_control": True,
    }
    any_audio_judge = gemini_enabled or openai_audio_enabled
    preview = None
    preview_public: dict[str, Any] = {}
    if any_audio_judge:
        _log(log_fp, "stage_start_audio_judge_preview", gemini_enabled=gemini_enabled, openai_audio_enabled=openai_audio_enabled)
        try:
            adaptive_preview_enabled = _env_on("BUSY_ADAPTIVE_AUDIO_JUDGE_PREVIEW", "1")
            if adaptive_preview_enabled:
                duration_sec = float(len(y) / max(int(sr), 1)) if y is not None else 0.0
                raw_preview_plan = build_adaptive_audio_preview_plan(
                    duration_sec=duration_sec,
                    features=features,
                    segment_analysis=features.get("segment_analysis", {}),
                    purpose="realtime_raw_audio_judge",
                    min_total_sec=float(os.environ.get("BUSY_AUDIO_PREVIEW_MIN_SEC", os.environ.get("BUSY_REALTIME_PREVIEW_MIN_SEC", "20")) or "20"),
                    base_total_sec=float(os.environ.get("BUSY_AUDIO_PREVIEW_BASE_SEC", os.environ.get("BUSY_REALTIME_PREVIEW_BASE_SEC", "30")) or "30"),
                    max_total_sec=float(os.environ.get("BUSY_AUDIO_PREVIEW_MAX_SEC", os.environ.get("BUSY_REALTIME_PREVIEW_MAX_SEC", "40")) or "40"),
                    default_segment_sec=float(os.environ.get("BUSY_AUDIO_PREVIEW_SEC", os.environ.get("BUSY_REALTIME_PREVIEW_SEC", "8")) or "8"),
                    max_segments=int(float(os.environ.get("BUSY_AUDIO_PREVIEW_MAX_SEGMENTS", os.environ.get("BUSY_REALTIME_PREVIEW_MAX_SEGMENTS", "5")) or "5")),
                )
                segments_req = int(raw_preview_plan.get("segment_count") or 4)
                segment_sec_req = float(raw_preview_plan.get("segment_sec") or 8.0)
                focus_segment_sec_req = float(raw_preview_plan.get("focus_segment_sec") or segment_sec_req)
                features["adaptive_audio_judge_preview_plan"] = raw_preview_plan
            else:
                raw_preview_plan = {"enabled": False, "reason": "BUSY_ADAPTIVE_AUDIO_JUDGE_PREVIEW disabled", "schema_version": "busy_adaptive_audio_preview_plan_v8_5_3_37"}
                segments_req = int(os.environ.get("BUSY_AUDIO_PREVIEW_SEGMENTS", os.environ.get("BUSY_REALTIME_PREVIEW_SEGMENTS", "3")) or "3")
                segment_sec_req = float(os.environ.get("BUSY_AUDIO_PREVIEW_SEC", os.environ.get("BUSY_REALTIME_PREVIEW_SEC", "10")) or "10")
                focus_segment_sec_req = float(os.environ.get("BUSY_AUDIO_PREVIEW_FOCUS_SEC", "20") or "20")
                features["adaptive_audio_judge_preview_plan"] = raw_preview_plan
            preview = build_realtime_genre_preview(
                y,
                sr,
                features.get("segment_analysis", {}),
                segments=segments_req,
                segment_sec=segment_sec_req,
                target_sr=int(os.environ.get("BUSY_AUDIO_PREVIEW_SR", os.environ.get("BUSY_REALTIME_PREVIEW_SR", "24000")) or "24000"),
                focus_segment_sec=focus_segment_sec_req,
            )
            preview_public = {k: v for k, v in preview.items() if k not in {"base64", "pcm16_base64"}}
            preview_public["adaptive_preview_plan"] = raw_preview_plan
            features["realtime_genre_preview"] = preview_public
            features["audio_judge_preview"] = preview_public
            _log(log_fp, "stage_done_audio_judge_preview", bytes=preview.get("bytes"), total_audio_sec=preview.get("total_audio_sec"), selected_segments=preview.get("selected_segments"), adaptive_target_sec=raw_preview_plan.get("target_total_sec"), adaptive_reasons=raw_preview_plan.get("risk_reasons"))
        except Exception as exc:
            preview = None
            preview_public = {"available": False, "reason": "preview_exception", "error": str(exc)[:500]}
            features["audio_judge_preview"] = preview_public
            _log(log_fp, "stage_error_audio_judge_preview", error=str(exc)[:500])

    audio_judges: dict[str, Any] = {}

    def _missing_or_disabled_audio_judges() -> None:
        if gemini_enabled and not preview:
            features["gemini_audio_judge"] = {"available": False, "reason": "missing_preview"}
            audio_judges["gemini_audio"] = features["gemini_audio_judge"]
        elif not gemini_enabled:
            features["gemini_audio_judge"] = {"available": False, "reason": "disabled"}

        if openai_audio_enabled and not preview:
            features["realtime_audio_judge"] = {"available": False, "reason": "missing_preview"}
            features["openai_realtime_audio_judge"] = features["realtime_audio_judge"]
            audio_judges["openai_realtime_audio"] = features["realtime_audio_judge"]
        elif not openai_audio_enabled:
            features["realtime_audio_judge"] = {"available": False, "reason": "disabled"}
            features["openai_realtime_audio_judge"] = features["realtime_audio_judge"]

    _missing_or_disabled_audio_judges()

    if preview and (gemini_enabled or openai_audio_enabled):
        parallel_audio_judges = (not ai_role_separation_enabled) and _env_on("BUSY_PARALLEL_AUDIO_JUDGES", "1") and gemini_enabled and openai_audio_enabled

        def _run_gemini_audio_judge() -> dict[str, Any]:
            _log(log_fp, "stage_start_gemini_audio_judge")
            gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("BUSY_GEMINI_API_KEY")
            return run_gemini_audio_genre_judge(
                api_key=gemini_key,
                preview_wav_base64=preview.get("base64"),
                preview_summary=preview_public,
                features=features,
                private_db=private_profile_db,
                model=os.environ.get("BUSY_GEMINI_AUDIO_MODEL", "gemini-2.5-flash-lite") or "gemini-2.5-flash-lite",
                mode=mode,
            )

        def _run_openai_audio_judge() -> dict[str, Any]:
            _log(log_fp, "stage_start_openai_realtime_audio_judge")
            return run_realtime_audio_genre_judge(
                api_key=os.environ.get("OPENAI_API_KEY"),
                preview_base64=preview.get("pcm16_base64") or preview.get("base64"),
                preview_summary=preview_public,
                features=features,
                private_db=private_profile_db,
                model=os.environ.get("BUSY_OPENAI_AUDIO_MODEL", os.environ.get("BUSY_REALTIME_GENRE_MODEL", "gpt-realtime-mini")) or "gpt-realtime-mini",
                mode=mode,
            )

        if parallel_audio_judges:
            _log(log_fp, "stage_start_parallel_audio_judges", judges=["gemini_flash_lite", "openai_realtime_mini"])
            with ThreadPoolExecutor(max_workers=2) as ex:
                future_map = {
                    "gemini": ex.submit(_run_gemini_audio_judge),
                    "openai": ex.submit(_run_openai_audio_judge),
                }
                for name, fut in future_map.items():
                    try:
                        res = fut.result()
                    except Exception as exc:
                        res = {"available": False, "reason": "audio_judge_exception", "error": str(exc)[:500]}
                    if name == "gemini":
                        features["gemini_audio_judge"] = res
                        audio_judges[_audio_judge_key(res, "gemini_audio")] = res
                        _log(log_fp, "stage_done_gemini_audio_judge", available=res.get("available"), model=res.get("model"), selected_profile=res.get("selected_profile"), confidence=res.get("confidence"), reason=res.get("reason"), error=res.get("error"))
                    else:
                        features["realtime_audio_judge"] = res
                        features["openai_realtime_audio_judge"] = res
                        audio_judges[_audio_judge_key(res, "openai_realtime_audio")] = res
                        _log(log_fp, "stage_done_openai_realtime_audio_judge", available=res.get("available"), model=res.get("model"), selected_profile=res.get("selected_profile"), selected_family=res.get("selected_family"), confidence=res.get("confidence"), reason=res.get("reason"), error=res.get("error"))
            _log(log_fp, "stage_done_parallel_audio_judges", available_count=sum(1 for v in audio_judges.values() if isinstance(v, dict) and v.get("available")))
        else:
            if gemini_enabled:
                try:
                    gemini_judge = _run_gemini_audio_judge()
                except Exception as exc:
                    gemini_judge = {"available": False, "reason": "audio_judge_exception", "error": str(exc)[:500]}
                features["gemini_audio_judge"] = gemini_judge
                audio_judges[_audio_judge_key(gemini_judge, "gemini_audio")] = gemini_judge
                _log(log_fp, "stage_done_gemini_audio_judge", available=gemini_judge.get("available"), model=gemini_judge.get("model"), selected_profile=gemini_judge.get("selected_profile"), confidence=gemini_judge.get("confidence"), reason=gemini_judge.get("reason"), error=gemini_judge.get("error"))
            if openai_audio_enabled:
                try:
                    rt_judge = _run_openai_audio_judge()
                except Exception as exc:
                    rt_judge = {"available": False, "reason": "audio_judge_exception", "error": str(exc)[:500]}
                features["realtime_audio_judge"] = rt_judge
                features["openai_realtime_audio_judge"] = rt_judge
                audio_judges[_audio_judge_key(rt_judge, "openai_realtime_audio")] = rt_judge
                _log(log_fp, "stage_done_openai_realtime_audio_judge", available=rt_judge.get("available"), model=rt_judge.get("model"), selected_profile=rt_judge.get("selected_profile"), selected_family=rt_judge.get("selected_family"), confidence=rt_judge.get("confidence"), reason=rt_judge.get("reason"), error=rt_judge.get("error"))

    features["audio_judges"] = audio_judges
    features["audio_judge_council"] = _audio_judge_council(audio_judges)
    _log(log_fp, "stage_done_audio_judge_council", available_count=features["audio_judge_council"].get("available_count"), primary_profiles=features["audio_judge_council"].get("primary_profiles"), overlap=features["audio_judge_council"].get("pairwise_genre_mix_overlap_first_two"))

    _log(log_fp, "stage_start_gpt_decision")
    api_key = os.environ.get("OPENAI_API_KEY")
    model_fast = os.environ.get("OPENAI_MODEL_FAST") or os.environ.get("OPENAI_MODEL") or "gpt-5.4-mini"
    high_accuracy_enabled = str(os.environ.get("BUSY_HIGH_ACCURACY_ESCALATION", "0")).strip().lower() in {"1", "true", "yes", "on"}
    model_high = (os.environ.get("OPENAI_MODEL_HIGH_ACCURACY") or "gpt-5.5") if high_accuracy_enabled else None
    model_router = os.environ.get("BUSY_MODEL_ROUTER") or ("smart" if high_accuracy_enabled else "fast_only")
    decision = build_mastering_decision(
        features,
        private_rules=private_rules,
        mode=mode,
        api_key=api_key,
        model=model_fast,
        model_high_accuracy=model_high,
        model_router=model_router,
        private_profile_db=private_profile_db,
    )
    _log(log_fp, "stage_done_gpt_decision", detected_style=decision.get("detected_style") or decision.get("genre_id"), confidence=decision.get("style_confidence"), boldness=decision.get("boldness_level"))

    # Attach pre-artifact stem evidence to the decision after GPT has consumed
    # it through features.  This keeps stage_state/debug report routing intact
    # without rerunning or rendering any Stage M audio.
    try:
        if isinstance(features.get("multi_component_assist"), dict):
            _attach_multicomponent_assist(features, decision, features.get("multi_component_assist") or {})
        sea = features.get("stem_evidence_assist") if isinstance(features.get("stem_evidence_assist"), dict) else {}
        if sea:
            decision.setdefault("input_assist_modes", {})["stem_evidence_assist"] = {
                "enabled": sea.get("enabled"),
                "provided": sea.get("provided"),
                "usable": sea.get("usable"),
                "global_reliability": sea.get("global_reliability"),
                "mode": sea.get("mode"),
                "audio_injection": False,
                "top_findings": sea.get("top_findings", [])[:8],
                "protection_priorities": sea.get("protection_priorities", {}),
                "residue_witness": sea.get("residue_witness", {}),
                "use_policy": sea.get("use_policy"),
            }
            decision.setdefault("final_decision_context", {})["stem_evidence_assist"] = decision["input_assist_modes"]["stem_evidence_assist"]
            flags = list(decision.get("processing_flags") or []) if isinstance(decision.get("processing_flags"), list) else []
            for flag in ["stem_evidence_assist_pre_artifact", "stage_m_replaced_by_stem_evidence_assist", "stem_audio_injection_forbidden"]:
                if flag not in flags:
                    flags.append(flag)
            decision["processing_flags"] = flags
    except Exception as exc:
        _log(log_fp, "stage_error_attach_stem_evidence_to_decision", error=str(exc)[:500])

    # v8.5.3.36: by default avoid spending a second Realtime/Gemini pass on the
    # raw/pre-decision preview. 5.4-mini receives the fused evidence and optional
    # QC-history strategy context; Gemini is saved for final-limited preview-master
    # listening before full render. Operators can still re-enable this old
    # conditional validator with BUSY_AUDIO_JUDGE_DECISION_VALIDATION=conditional.
    validation_mode = str(os.environ.get("BUSY_AUDIO_JUDGE_DECISION_VALIDATION", "disabled") or "disabled").strip().lower()
    validation_disabled = validation_mode in {"0", "false", "off", "no", "disabled"}
    validation_always = validation_mode in {"1", "true", "yes", "on", "always"}
    validation_triggers = _post_decision_validation_triggers(features, decision)
    decision.setdefault("final_arbitration", {})["post_decision_validation_triggers"] = validation_triggers
    decision.setdefault("final_decision_context", {})["post_decision_validation_triggers"] = validation_triggers
    should_validate = (not validation_disabled) and preview and (validation_always or bool(validation_triggers.get("enabled")))
    if should_validate:
        _log(log_fp, "stage_start_conditional_post_decision_validation", mode=validation_mode, reasons=validation_triggers.get("reasons"))
        validators: dict[str, Any] = {}
        gemini_validation_enabled = _env_on("BUSY_GEMINI_AUDIO_VALIDATION", os.environ.get("BUSY_GEMINI_AUDIO_JUDGE", "1"))
        openai_validation_default = "0" if ai_role_separation_enabled else os.environ.get("BUSY_OPENAI_AUDIO_JUDGE", "1")
        openai_validation_enabled = _env_on("BUSY_OPENAI_AUDIO_VALIDATION", openai_validation_default)

        def _run_gemini_validator() -> dict[str, Any]:
            _log(log_fp, "stage_start_gemini_post_decision_validation")
            gemini_key_v = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("BUSY_GEMINI_API_KEY")
            return run_gemini_audio_decision_validator(
                api_key=gemini_key_v,
                preview_wav_base64=preview.get("base64"),
                preview_summary=preview_public,
                features=features,
                decision=decision,
                private_db=private_profile_db,
                model=os.environ.get("BUSY_GEMINI_AUDIO_VALIDATION_MODEL", os.environ.get("BUSY_GEMINI_AUDIO_MODEL", "gemini-2.5-flash-lite")) or "gemini-2.5-flash-lite",
                mode=mode,
            )

        def _run_openai_validator() -> dict[str, Any]:
            _log(log_fp, "stage_start_openai_realtime_post_decision_validation")
            return run_realtime_audio_decision_validator(
                api_key=os.environ.get("OPENAI_API_KEY"),
                preview_base64=preview.get("pcm16_base64") or preview.get("base64"),
                preview_summary=preview_public,
                features=features,
                decision=decision,
                private_db=private_profile_db,
                model=os.environ.get("BUSY_OPENAI_AUDIO_VALIDATION_MODEL", os.environ.get("BUSY_OPENAI_AUDIO_MODEL", "gpt-realtime-mini")) or "gpt-realtime-mini",
                mode=mode,
            )

        parallel_validators = (not ai_role_separation_enabled) and _env_on("BUSY_PARALLEL_DECISION_VALIDATORS", "1") and gemini_validation_enabled and openai_validation_enabled
        if parallel_validators:
            _log(log_fp, "stage_start_parallel_post_decision_validation", validators=["gemini_flash_lite", "openai_realtime_mini"])
            with ThreadPoolExecutor(max_workers=2) as ex:
                future_map = {
                    "gemini_flash_lite": ex.submit(_run_gemini_validator),
                    "openai_realtime_mini": ex.submit(_run_openai_validator),
                }
                for key, fut in future_map.items():
                    try:
                        validators[key] = fut.result()
                    except Exception as exc:
                        validators[key] = {"available": False, "reason": "post_decision_validation_exception", "error": str(exc)[:500]}
                    res = validators[key]
                    if key == "gemini_flash_lite":
                        _log(log_fp, "stage_done_gemini_post_decision_validation", available=res.get("available"), verdict=res.get("verdict"), support_score=res.get("support_score"), reason=res.get("reason"), error=res.get("error"))
                    else:
                        _log(log_fp, "stage_done_openai_realtime_post_decision_validation", available=res.get("available"), verdict=res.get("verdict"), support_score=res.get("support_score"), reason=res.get("reason"), error=res.get("error"))
            _log(log_fp, "stage_done_parallel_post_decision_validation", available_count=sum(1 for v in validators.values() if isinstance(v, dict) and v.get("available")))
        else:
            if gemini_validation_enabled:
                try:
                    gv = _run_gemini_validator()
                except Exception as exc:
                    gv = {"available": False, "reason": "post_decision_validation_exception", "error": str(exc)[:500]}
                validators["gemini_flash_lite"] = gv
                _log(log_fp, "stage_done_gemini_post_decision_validation", available=gv.get("available"), verdict=gv.get("verdict"), support_score=gv.get("support_score"), reason=gv.get("reason"), error=gv.get("error"))
            if openai_validation_enabled:
                try:
                    rv = _run_openai_validator()
                except Exception as exc:
                    rv = {"available": False, "reason": "post_decision_validation_exception", "error": str(exc)[:500]}
                validators["openai_realtime_mini"] = rv
                _log(log_fp, "stage_done_openai_realtime_post_decision_validation", available=rv.get("available"), verdict=rv.get("verdict"), support_score=rv.get("support_score"), reason=rv.get("reason"), error=rv.get("error"))
        council_v = _post_decision_validation_council(validators)
        council_v["trigger_diagnostics"] = validation_triggers
        features["post_decision_audio_validation"] = {**validators, "council": council_v}
        decision.setdefault("final_arbitration", {})["post_decision_audio_validation"] = council_v
        decision.setdefault("final_decision_context", {})["post_decision_audio_validation"] = council_v
        if council_v.get("requires_revision"):
            decision.setdefault("processing_flags", []).append("post_decision_audio_validation_requires_revision")
        _log(log_fp, "stage_done_post_decision_audio_validation_council", available_count=council_v.get("available_count"), verdicts=council_v.get("verdicts"), average_support_score=council_v.get("average_support_score"), requires_revision=council_v.get("requires_revision"))

        # One 5.4-mini repair pass after validation, before any 5.5 arbitration.
        if council_v.get("requires_revision") and _env_on("BUSY_POST_VALIDATION_54_REVISION", "0"):
            _log(log_fp, "stage_start_post_validation_54_revision")
            try:
                revised = build_mastering_decision(
                    features,
                    private_rules=private_rules,
                    mode=mode,
                    api_key=api_key,
                    model=model_fast,
                    model_high_accuracy=None,
                    model_router="fast_only",
                    private_profile_db=private_profile_db,
                )
                revised.setdefault("final_arbitration", {})["post_decision_audio_validation"] = council_v
                revised.setdefault("final_decision_context", {})["post_decision_audio_validation"] = council_v
                revised["post_validation_revision"] = {"enabled": True, "model": model_fast, "reason": "validation_requires_revision", "previous_selected_profile": decision.get("selected_profile"), "previous_genre_mix": decision.get("genre_mix")}
                decision = revised
                _log(log_fp, "stage_done_post_validation_54_revision", selected_profile=decision.get("selected_profile"), model=decision.get("model_used"))
            except Exception as exc:
                decision.setdefault("notes", []).append(f"Post-validation 5.4 revision failed; previous decision kept: {exc}")
                decision.setdefault("post_validation_revision", {})["error"] = str(exc)[:500]
                _log(log_fp, "stage_error_post_validation_54_revision", error=str(exc)[:500])

        needs_high, high_reasons = _post_validation_needs_high_arbitration(council_v)
        if needs_high and _env_on("BUSY_POST_VALIDATION_HIGH_ARBITRATION", "1"):
            high_model = os.environ.get("OPENAI_MODEL_HIGH_ACCURACY") or "gpt-5.5"
            _log(log_fp, "stage_start_post_validation_high_arbitration", model=high_model, reasons=high_reasons)
            try:
                high_decision = build_mastering_decision(
                    features,
                    private_rules=private_rules,
                    mode=mode,
                    api_key=api_key,
                    model=high_model,
                    model_high_accuracy=None,
                    model_router="high_accuracy_only",
                    private_profile_db=private_profile_db,
                )
                high_decision.setdefault("final_arbitration", {})["post_decision_audio_validation"] = council_v
                high_decision.setdefault("final_decision_context", {})["post_decision_audio_validation"] = council_v
                high_decision["post_validation_high_arbitration"] = {"enabled": True, "model": high_model, "reasons": high_reasons, "previous_selected_profile": decision.get("selected_profile"), "previous_genre_mix": decision.get("genre_mix")}
                high_decision["high_accuracy_escalated"] = True
                high_decision["high_accuracy_reasons"] = high_reasons
                decision = high_decision
                _log(log_fp, "stage_done_post_validation_high_arbitration", selected_profile=decision.get("selected_profile"), model=decision.get("model_used"))
            except Exception as exc:
                decision.setdefault("notes", []).append(f"Post-validation high arbitration failed; previous decision kept: {exc}")
                decision.setdefault("post_validation_high_arbitration", {})["error"] = str(exc)[:500]
                _log(log_fp, "stage_error_post_validation_high_arbitration", error=str(exc)[:500])
    else:
        reason = "disabled" if validation_disabled else ("missing_preview" if not preview else "conditional_triggers_not_met")
        features["post_decision_audio_validation"] = {"enabled": False, "reason": reason, "mode": validation_mode, "trigger_diagnostics": validation_triggers}
        decision.setdefault("final_arbitration", {})["post_decision_audio_validation"] = features["post_decision_audio_validation"]
        decision.setdefault("final_decision_context", {})["post_decision_audio_validation"] = features["post_decision_audio_validation"]
        _log(log_fp, "stage_skip_post_decision_audio_validation", reason=reason, mode=validation_mode, triggers=validation_triggers.get("reasons"))

    if private_profile_is_v7:
        _log(log_fp, "stage_start_v7_profile_overlay")
        decision = build_v7_overlay(features, decision, private_profile_db, mode=mode)
        rv7 = decision.get("reference_v7", {})
        _log(log_fp, "stage_done_v7_profile_overlay", selected_profile=rv7.get("selected_profile"), family=rv7.get("family"), delivery_mode=rv7.get("delivery_mode"), repair_active=rv7.get("repair_active"), selection_source=rv7.get("selection_source"))

    decision = _canonicalize_decision_for_stage(decision, stage="A-1_decision_finalized")
    _log(log_fp, "stage_done_decision_canonicalized", **_decision_profile_digest(decision))

    if _env_on("BUSY_ROLE_FIRST_DSP_CONTRACT", "1"):
        _log(log_fp, "stage_start_role_first_dsp_contract")
        decision = apply_role_first_dsp_contract(features, decision, private_profile_db=private_profile_db, mode=mode)
        try:
            features["role_first_dsp_contract"] = decision.get("role_first_dsp_contract", {})
        except Exception:
            pass
        _log(
            log_fp,
            "stage_done_role_first_dsp_contract",
            enabled=bool((decision.get("role_first_dsp_contract") or {}).get("enabled") if isinstance(decision.get("role_first_dsp_contract"), dict) else False),
            actual_dsp_basis=decision.get("actual_dsp_basis"),
            selected_profile=decision.get("selected_profile"),
            role_count=len((((decision.get("role_first_dsp_contract") or {}).get("role_map") or {}).get("role_scores") or {}) if isinstance(decision.get("role_first_dsp_contract"), dict) else {}),
        )

    if _env_on("BUSY_EXISTING_PROCESS_RESEARCH_GOVERNANCE", "1"):
        try:
            decision = _attach_existing_process_research_governance(features, decision, stage="A36_A39_existing_decision_governance")
            _log(
                log_fp,
                "stage_done_existing_process_research_governance",
                preserve_count=len(((decision.get("existing_process_research_governance") or {}).get("preserve_priorities") or []) if isinstance(decision.get("existing_process_research_governance"), dict) else []),
                control_count=len(((decision.get("existing_process_research_governance") or {}).get("control_priorities") or []) if isinstance(decision.get("existing_process_research_governance"), dict) else []),
                reduce_count=len(((decision.get("existing_process_research_governance") or {}).get("reduce_priorities") or []) if isinstance(decision.get("existing_process_research_governance"), dict) else []),
            )
        except Exception as exc:
            features["existing_process_research_governance"] = {"active": False, "reason": "exception", "error": str(exc)[:500]}
            _log(log_fp, "stage_error_existing_process_research_governance", error=str(exc)[:500])

    # v8.5.3.10: Frequency-Band Centrifuge residue intelligence.
    # This runs after detailed profile/role context exists, so intentional
    # texture guards can protect hard-techno rumble, phonk cowbells, vocal air,
    # lo-fi hiss, live ambience, etc.  v1 is analysis-only and only emits
    # indirect mastering influence hints unless explicitly changed later.
    if _env_on("BUSY_RESIDUE_CENTRIFUGE", "1"):
        _log(log_fp, "stage_start_residue_centrifuge")
        try:
            residue_source = y_analysis if y_analysis is not None else y
            residue_analysis = analyze_frequency_band_centrifuge(residue_source, sr, decision=decision, pre_analysis=features)
            features["frequency_band_centrifuge"] = residue_analysis
            decision.setdefault("residue_centrifuge", {
                "enabled": bool(residue_analysis.get("enabled")),
                "active": bool(residue_analysis.get("active")),
                "global_summary": residue_analysis.get("global_summary", {}),
                "indirect_mastering_influence": residue_analysis.get("indirect_mastering_influence", {}),
                "policy": "analysis_only_input_evidence; mastering may use indirect high-band residue guardrails",
            })
            _log(
                log_fp,
                "stage_done_residue_centrifuge",
                active=residue_analysis.get("active"),
                mode=residue_analysis.get("mode"),
                high_band_residue_risk=(residue_analysis.get("global_summary") or {}).get("high_band_residue_risk") if isinstance(residue_analysis.get("global_summary"), dict) else None,
                side_high_residue_risk=(residue_analysis.get("global_summary") or {}).get("side_high_residue_risk") if isinstance(residue_analysis.get("global_summary"), dict) else None,
                hotspot_count=len(residue_analysis.get("hotspot_events") or []) if isinstance(residue_analysis, dict) else 0,
                action_count=len(residue_analysis.get("proposed_actions") or []) if isinstance(residue_analysis, dict) else 0,
            )
        except Exception as exc:
            features["frequency_band_centrifuge"] = {"enabled": True, "active": False, "reason": "exception", "error_type": type(exc).__name__, "error": str(exc)[:500]}
            decision.setdefault("residue_centrifuge", features["frequency_band_centrifuge"])
            _log(log_fp, "stage_error_residue_centrifuge", error=str(exc)[:500])


    # v8.5.3.62.9.9.2: persist v62.9.9 governance in stage_state before
    # Stage B loads the decision.  This makes the five research-governance
    # payloads visible and actionable in the fresh limiter worker.
    try:
        decision = _attach_v6299_research_governance_if_missing(features, decision, mode=mode, stage="A40_after_residue_before_mix_conditioning", log_fp=log_fp)
    except Exception as exc:
        features["v62_9_9_research_governance"] = {"enabled": True, "active": False, "reason": "exception", "error": str(exc)[:500]}
        _log(log_fp, "stage_error_v6299_research_governance", error=str(exc)[:500])

    # v8.5.3.40: stemless Mix Conditioning planning layer.
    # This is not a stem-based remix and it does not write a conditioned WAV.
    # It evaluates the post-pre-clean stereo mixdown for masterability, builds
    # bounded conditioning actions/risk caps, and folds them into the composed
    # final render chain before blocker remediation and final intensity.
    try:
        mix_conditioning_plan = build_mix_conditioning_plan(features, decision)
        features["mix_conditioning_plan"] = mix_conditioning_plan
        features["mix_conditioning_analysis"] = mix_conditioning_plan.get("analysis", {}) if isinstance(mix_conditioning_plan, dict) else {}
        features["mix_conditioning_validation"] = mix_conditioning_plan.get("validation", {}) if isinstance(mix_conditioning_plan, dict) else {}
        features["mix_conditioning_actions"] = mix_conditioning_plan.get("actions", []) if isinstance(mix_conditioning_plan, dict) else []
        features["mix_conditioning_feature_deltas"] = mix_conditioning_plan.get("predicted_feature_deltas", {}) if isinstance(mix_conditioning_plan, dict) else {}
        features["loudness_feasibility_predictor"] = mix_conditioning_plan.get("loudness_feasibility", {}) if isinstance(mix_conditioning_plan, dict) else build_loudness_feasibility_score(features, decision)
        features["perceptual_quality_score_baseline"] = calculate_perceptual_quality_score(features)
        features["multi_criteria_stress_window_requirements"] = mix_conditioning_plan.get("stress_window_requirements", {}) if isinstance(mix_conditioning_plan, dict) else {}
        decision = apply_mix_conditioning_to_decision(decision, mix_conditioning_plan)
        _log(
            log_fp,
            "stage_done_mix_conditioning_plan",
            active=bool(mix_conditioning_plan.get("active")) if isinstance(mix_conditioning_plan, dict) else False,
            need_count=len(((mix_conditioning_plan.get("analysis") or {}).get("detected_needs") or []) if isinstance(mix_conditioning_plan, dict) else []),
            action_count=len(mix_conditioning_plan.get("actions") or []) if isinstance(mix_conditioning_plan, dict) else 0,
            feasibility_score=((mix_conditioning_plan.get("loudness_feasibility") or {}).get("score") if isinstance(mix_conditioning_plan, dict) else None),
            recommended_max_intensity=((mix_conditioning_plan.get("loudness_feasibility") or {}).get("recommended_max_intensity") if isinstance(mix_conditioning_plan, dict) else None),
            intermediate_wav_written=False,
        )
    except Exception as exc:
        features["mix_conditioning_plan"] = {"enabled": True, "active": False, "reason": "exception", "error": str(exc)[:500]}
        features["loudness_feasibility_predictor"] = build_loudness_feasibility_score(features, decision)
        decision["mix_conditioning_plan"] = features["mix_conditioning_plan"]
        decision["loudness_feasibility_predictor"] = features["loudness_feasibility_predictor"]
        _log(log_fp, "stage_error_mix_conditioning_plan", error=str(exc)[:500])

    # v8.5.3.34/v8.5.3.40: decide final intensity after conditioning + blocker-remediation planning.
    # This preserves the existing intensity policy, but changes the evidence
    # basis from raw blockers to post-pre-clean / mix-conditioning / post-remediation
    # predicted metrics.  No conditioned/remediated premaster WAV is written here.
    try:
        blocker_plan = build_blocker_remediation_plan(features, decision)
        features["blocker_remediation_plan"] = blocker_plan
        features["post_remediation_predicted_metrics"] = blocker_plan.get("post_remediation_predicted_metrics", {})
        decision["blocker_remediation_plan"] = blocker_plan
        decision["post_remediation_predicted_metrics"] = blocker_plan.get("post_remediation_predicted_metrics", {})
        decision["optimized_processing_order"] = build_pipeline_architecture_state(features, decision, final_intensity=None)
        _log(
            log_fp,
            "stage_done_blocker_remediation_plan",
            blocker_count=len(blocker_plan.get("blockers") or []),
            action_count=len(blocker_plan.get("actions") or []),
            intermediate_wav_written=False,
            raw_artifact=(blocker_plan.get("raw_blocker_metrics") or {}).get("artifact_score"),
            post_artifact=(blocker_plan.get("post_remediation_predicted_metrics") or {}).get("artifact_score"),
        )
    except Exception as exc:
        features["blocker_remediation_plan"] = {"enabled": False, "reason": "exception", "error": str(exc)[:500]}
        decision["blocker_remediation_plan"] = features["blocker_remediation_plan"]
        _log(log_fp, "stage_error_blocker_remediation_plan", error=str(exc)[:500])

    # v8.5.3.63.6.0.1: research capability framework / remaining capability matrix debug hotfix.
    # This is a planning/telemetry layer only.  It deliberately runs after
    # mix-conditioning and blocker-remediation planning but before final
    # intensity ownership so the router sees current blockers without forcing
    # any new DSP.
    features, decision, _v6360_fw = _attach_v6360_research_capability_framework_if_missing(
        features,
        decision,
        stage="A40_after_blocker_before_auto_intensity",
        log_fp=log_fp,
    )

    _resolve_auto_intensity_policy(features, decision, os.environ.get("BUSY_AUTO_MASTERING_INTENSITY_REQUESTED") or os.environ.get("BUSY_MASTERING_INTENSITY") or "auto", log_fp, phase="A-1_final_decision")
    try:
        final_intensity_for_graph = decision.get("mastering_intensity") or (decision.get("auto_intensity_policy") or {}).get("final_mode")
        graph = build_pipeline_architecture_state(features, decision, final_intensity=final_intensity_for_graph)
        features["optimized_pipeline_architecture"] = graph
        decision["optimized_processing_order"] = graph
        decision["render_budget_policy"] = render_budget_policy(final_intensity_for_graph)
        _log(log_fp, "stage_done_optimized_processing_graph_bound", final_intensity=final_intensity_for_graph, hard_full_render_ceiling=(decision.get("render_budget_policy") or {}).get("hard_full_render_ceiling"), intermediate_wav_policy=graph.get("intermediate_wav_policy"))
    except Exception as exc:
        _log(log_fp, "stage_error_optimized_processing_graph_bound", error=str(exc)[:500])

    private_bundle = None
    private_json = None
    private_rules = ""
    private_profile_db = None
    gc.collect()
    return y, sr, features, decision, preflight_report


def _download_optional_multicomponent_zip(args: argparse.Namespace, work_dir: Path, log_fp) -> Path | None:
    """Download optional Suno separated-track ZIP for analysis-only multi-component assist."""
    zip_path_arg = str(getattr(args, "stem_zip", "") or "").strip()
    supabase_path = str(getattr(args, "supabase_stem_zip", "") or "").strip()
    if not zip_path_arg and not supabase_path:
        return None
    local_zip = Path(zip_path_arg) if zip_path_arg else work_dir / "suno_separated_tracks.zip"
    if zip_path_arg and local_zip.exists():
        _log(log_fp, "multi_component_assist_zip_local_ready", path=str(local_zip), bytes=local_zip.stat().st_size)
        return local_zip
    if supabase_path:
        try:
            _log(log_fp, "stage_start_multicomponent_zip_download", path=supabase_path)
            storage = SupabaseStorage.from_env()
            storage.download_file(supabase_path, str(local_zip))
            _log(log_fp, "stage_done_multicomponent_zip_download", local=str(local_zip), bytes=local_zip.stat().st_size)
            return local_zip
        except Exception as exc:
            _log(log_fp, "multi_component_assist_zip_download_failed", path=supabase_path, error=str(exc)[:500])
            return None
    return None


def _build_stage_m_reference_profile(features: dict[str, Any] | None, decision: dict[str, Any] | None) -> dict[str, Any]:
    """Adapt the already-built original stereo analysis/decision package for Stage M.

    This does not run a second full analysis. It only normalizes values that
    already exist in features/decision so the Stage M stem mixdown builder can
    treat the original stereo as the musical intent reference.
    """
    features = features if isinstance(features, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    analysis = features.get("analysis") if isinstance(features.get("analysis"), dict) else features
    seg = features.get("segments") or features.get("segment_analysis") or features.get("segment_summary") or {}
    mix_plan = features.get("mix_conditioning_plan") if isinstance(features.get("mix_conditioning_plan"), dict) else {}
    v58 = features.get("v58_spectral_conditioning_plan") if isinstance(features.get("v58_spectral_conditioning_plan"), dict) else {}
    v59 = features.get("v59_lf_crest_source_conditioning_plan") if isinstance(features.get("v59_lf_crest_source_conditioning_plan"), dict) else {}
    blockers = []
    for src in [
        (mix_plan.get("analysis") or {}).get("detected_needs") if isinstance(mix_plan, dict) else None,
        mix_plan.get("blockers") if isinstance(mix_plan, dict) else None,
        features.get("v61_remaining_blockers"),
        (features.get("v61_render_measured_feedback_optimizer") or {}).get("remaining_blockers") if isinstance(features.get("v61_render_measured_feedback_optimizer"), dict) else None,
    ]:
        if isinstance(src, list):
            blockers.extend(str(x) for x in src if x)
    # Infer common blocker tags from existing v58/v59/v57 state when compact names are absent.
    try:
        if isinstance(v58, dict) and (v58.get("active") or v58.get("low_mid_bottleneck")):
            blockers.append("low_mid_bottleneck")
    except Exception:
        pass
    try:
        if isinstance(v59, dict) and v59.get("active"):
            blockers.append("lf_crest_source_conditioning_needed")
    except Exception:
        pass
    def _stage_m_hint_text(obj, limit=8000):
        parts = []
        def walk(o, depth=0):
            if len(" ".join(parts)) > limit or depth > 5:
                return
            if isinstance(o, str):
                if o.strip(): parts.append(o.strip())
            elif isinstance(o, dict):
                for k, v in list(o.items())[:120]:
                    ks = str(k).lower()
                    if any(tok in ks for tok in ["judge", "gemini", "realtime", "listener", "improvement", "target", "genre", "profile", "blocker", "issue", "hint", "evidence"]):
                        parts.append(str(k))
                        walk(v, depth + 1)
                    elif depth < 2:
                        walk(v, depth + 1)
            elif isinstance(o, (list, tuple, set)):
                for v in list(o)[:120]: walk(v, depth + 1)
        walk(obj)
        return " ".join(parts)[:limit]
    evidence_hint_text = _stage_m_hint_text({"features": features, "decision": decision})
    ai_evidence_hints = []
    for tok in ["low-end punch", "drop density", "transient punch", "vocal clarity", "club", "EDM", "lra", "crest", "true peak", "low-mid"]:
        if tok.lower() in evidence_hint_text.lower():
            ai_evidence_hints.append(tok)

    profile = {
        "schema_version": "busy_stage_m_reference_intent_adapter_v8_5_3_62_5_3",
        "source": "existing_original_features_and_decision",
        "no_second_full_original_analysis": True,
        "selected_profile": decision.get("selected_profile") or decision.get("profile") or decision.get("genre_profile"),
        "profile_blend": decision.get("profile_blend") or decision.get("genre_profile_blend"),
        "genre_profile": {
            "genre": decision.get("genre") or decision.get("selected_genre") or decision.get("style"),
            "subgenre": decision.get("subgenre") or decision.get("selected_subgenre"),
            "profile": decision.get("selected_profile") or decision.get("profile"),
            "source": "decision_fusion",
        },
        "original_metrics": {
            "integrated_lufs": analysis.get("integrated_lufs") or analysis.get("lufs") or features.get("input_lufs"),
            "true_peak_dbtp": analysis.get("true_peak_dbtp") or analysis.get("approx_true_peak_dbfs") or features.get("true_peak_dbtp"),
            "lra_lu": analysis.get("lra_lu") or features.get("lra_lu"),
            "crest_factor_db": analysis.get("crest_factor_db") or features.get("crest_factor_db"),
            "bpm": features.get("bpm") or decision.get("bpm"),
        },
        "segment_summary": seg if isinstance(seg, dict) else {"available": bool(seg)},
        "mastering_blockers": sorted(set(blockers)),
        "evidence_hint_text": evidence_hint_text,
        "ai_evidence_hints": sorted(set(ai_evidence_hints)),
        "stage_m_use": "evidence_to_action_target_zones_action_specific_safety_and_validation",
    }
    return profile


def _analyze_optional_multicomponent_assist(local_input: Path, local_zip: Path | None, work_dir: Path, log_fp, premaster_output_path: Path | None = None, reference_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Deprecated late stem path kept as a safe fallback.

    v8.5.3.62.9.7 replaces Stage M stem mixdown with the earlier Stem Evidence
    Assist path.  This function must never call the Stage M mixdown builder and
    must never write/select a stem-derived premaster.  If it is reached by a
    legacy path, it runs analysis-only multicomponent assist and returns the
    original-stereo route.
    """
    premaster_output_path = None  # hard-disable legacy residual/injection premaster
    if local_zip is None:
        result = {
            "schema_version": "busy_multi_component_assist_v8_5_3_62_9_7_stem_evidence_stage_m_removed",
            "provided": False,
            "enabled": _env_on("BUSY_MULTI_COMPONENT_ASSIST", "1"),
            "usable": False,
            "recommended_mode": "not_provided_stereo_only",
            "separated_tracks_used_for_audio_render": False,
            "downstream_multicomponent_assist_bypassed": True,
            "downstream_bypass_reason": "stem_zip_not_provided",
            "stage_m_mixdown_builder": {"enabled": False, "attempted": False, "accepted": False, "reason": "stage_m_removed_stem_zip_not_provided"},
            "stage_m_mixdown_candidate": {"accepted": False, "reason": "stage_m_removed"},
        }
        result["stage_m_stem_premaster_builder"] = _build_stage_m_stem_premaster_report(result, stem_zip_provided=False)
        return result

    result: dict[str, Any]
    try:
        _log(log_fp, "stage_start_stem_evidence_assist_legacy_fallback", zip=str(local_zip), stage_m_mixdown_removed=True)
        result = analyze_multicomponent_assist(local_input, local_zip, work_dir=work_dir, premaster_output_path=None)
        if isinstance(result, dict):
            result.pop("stem_assisted_premaster_optimization", None)
        _log(
            log_fp,
            "stage_done_stem_evidence_assist_legacy_fallback",
            provided=result.get("provided"),
            usable=result.get("usable"),
            recommended_mode=result.get("recommended_mode"),
            global_grade=((result.get("global_reconstruction") or {}).get("global_reconstruction_grade") if isinstance(result.get("global_reconstruction"), dict) else None),
            tracks=result.get("track_count_analyzed"),
            stage_m_mixdown_called=False,
            audio_render=False,
        )
    except Exception as exc:
        _log(log_fp, "stem_evidence_assist_legacy_fallback_failed", error=str(exc)[:500])
        result = {
            "schema_version": "busy_multi_component_assist_v8_5_3_62_9_7_stem_evidence_stage_m_removed",
            "provided": True,
            "enabled": True,
            "usable": False,
            "recommended_mode": "analysis_failed_stereo_only_fallback",
            "error": str(exc)[:500],
            "separated_tracks_used_for_audio_render": False,
        }
    result["schema_version"] = str(result.get("schema_version") or "busy_multi_component_assist_v8_5_3_46") + "_stage_m_removed"
    result["recommended_mode"] = "stem_evidence_assist_analysis_only"
    result["separated_tracks_used_for_audio_render"] = False
    result["stage_m_mixdown_builder"] = {"enabled": False, "attempted": False, "accepted": False, "reason": "stage_m_replaced_by_stem_evidence_assist"}
    result["stage_m_mixdown_candidate"] = {"accepted": False, "reason": "stage_m_replaced_by_stem_evidence_assist"}
    result["stage_m_stem_premaster_builder"] = _build_stage_m_stem_premaster_report(result, stem_zip_provided=True)
    result["downstream_multicomponent_assist_bypassed"] = False
    result["downstream_bypass_reason"] = "original_stereo_route_keeps_stem_evidence_assist"
    return result

def _attach_multicomponent_assist(features: dict[str, Any], decision: dict[str, Any], assist: dict[str, Any]) -> None:
    if not isinstance(features, dict) or not isinstance(assist, dict):
        return
    features["multi_component_assist"] = assist
    lgcca = assist.get("low_grs_clean_component_assist") if isinstance(assist.get("low_grs_clean_component_assist"), dict) else {}
    ssrw = assist.get("stem_sum_residue_witness") if isinstance(assist.get("stem_sum_residue_witness"), dict) else {}
    features["low_grs_clean_component_assist"] = lgcca
    features["stem_sum_residue_witness"] = ssrw
    stage_m_builder = assist.get("stage_m_mixdown_builder") if isinstance(assist.get("stage_m_mixdown_builder"), dict) else {}
    features["multi_component_assist_summary"] = {
        "provided": assist.get("provided"),
        "usable": assist.get("usable"),
        "recommended_mode": assist.get("recommended_mode"),
        "global_reconstruction_grade": ((assist.get("global_reconstruction") or {}).get("global_reconstruction_grade") if isinstance(assist.get("global_reconstruction"), dict) else None),
        "track_count_analyzed": assist.get("track_count_analyzed"),
        "stage_m_mixdown_builder_present": bool(stage_m_builder and stage_m_builder.get("attempted")),
        "stage_m_mixdown_accepted": stage_m_builder.get("accepted") if stage_m_builder else None,
        "stage_m_mixdown_reason": stage_m_builder.get("reason") if stage_m_builder else None,
        "stage_m_removed": str(stage_m_builder.get("reason") or "").startswith("stage_m_replaced") if stage_m_builder else True,
        "stage_m_usable_stem_count": stage_m_builder.get("usable_stem_count") if stage_m_builder else None,
        "stage_m_avg_reliability_score": stage_m_builder.get("avg_reliability_score") if stage_m_builder else None,
        "legacy_stem_assisted_premaster_removed": True,
        "legacy_stem_assisted_premaster_not_called": True,
        "lgcca_enabled": lgcca.get("enabled") if isinstance(lgcca, dict) else None,
        "lgcca_active": lgcca.get("active") if isinstance(lgcca, dict) else None,
        "lgcca_mode_selected": lgcca.get("mode_selected") if isinstance(lgcca, dict) else None,
        "lgcca_audio_injection_forbidden": lgcca.get("audio_injection_forbidden", True) if isinstance(lgcca, dict) else True,
        "lgcca_control_map_summary": lgcca.get("control_map_summary", {}) if isinstance(lgcca, dict) else {},
        "lgcca_evidence_gate_summary": lgcca.get("evidence_gate_summary", {}) if isinstance(lgcca, dict) else {},
        "stem_sum_residue_witness": compact_ssrw_summary(ssrw),
        "gemini_focus_points": ((assist.get("analysis_summary") or {}).get("gemini_focus_points") if isinstance(assist.get("analysis_summary"), dict) else []),
    }
    decision.setdefault("input_assist_modes", {})["multi_component_assist"] = features["multi_component_assist_summary"]
    if ssrw:
        decision.setdefault("input_assist_modes", {})["stem_sum_residue_witness"] = compact_ssrw_summary(ssrw)
        decision.setdefault("final_decision_context", {})["stem_sum_residue_witness"] = compact_ssrw_summary(ssrw)
    if lgcca:
        decision.setdefault("input_assist_modes", {})["low_grs_clean_component_assist"] = {
            "enabled": lgcca.get("enabled"),
            "active": lgcca.get("active"),
            "mode_selected": lgcca.get("mode_selected"),
            "audio_injection_forbidden": lgcca.get("audio_injection_forbidden", True),
            "control_map_summary": lgcca.get("control_map_summary", {}),
            "candidate_scoring_hints": lgcca.get("candidate_scoring_hints", {}),
            "limiter_risk_hints": lgcca.get("limiter_risk_hints", {}),
        }
        decision.setdefault("final_decision_context", {})["low_grs_clean_component_assist"] = decision["input_assist_modes"]["low_grs_clean_component_assist"]
    flags = list(decision.get("processing_flags") or []) if isinstance(decision.get("processing_flags"), list) else []
    if assist.get("provided") and "multi_component_assist_zip_provided" not in flags:
        flags.append("multi_component_assist_zip_provided")
    if assist.get("usable") and "multi_component_assist_analysis_available" not in flags:
        flags.append("multi_component_assist_analysis_available")
    if lgcca.get("active") and "low_grs_clean_component_assist_active" not in flags:
        flags.append("low_grs_clean_component_assist_active")
    if lgcca.get("audio_injection_forbidden", False) and "stem_audio_injection_forbidden" not in flags:
        flags.append("stem_audio_injection_forbidden")
    if ssrw.get("enabled") and "stem_sum_residue_witness_available" not in flags:
        flags.append("stem_sum_residue_witness_available")
    if ssrw.get("no_injection_verified", True) and "ssrw_no_injection_verified" not in flags:
        flags.append("ssrw_no_injection_verified")
    decision["processing_flags"] = flags



def _stage_m_avg_track_trust(assist: dict[str, Any]) -> float:
    tracks = assist.get("track_trust_matrix") if isinstance(assist, dict) else []
    vals: list[float] = []
    if isinstance(tracks, list):
        for t in tracks:
            if not isinstance(t, dict):
                continue
            try:
                vals.append(float(t.get("track_trust_score") or 0.0))
            except Exception:
                pass
    return round(float(sum(vals) / len(vals)), 3) if vals else 0.0


def _stage_m_role_compact(assist: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    tracks = assist.get("track_trust_matrix") if isinstance(assist, dict) else []
    out: list[dict[str, Any]] = []
    if not isinstance(tracks, list):
        return out
    for t in tracks[:limit]:
        if not isinstance(t, dict):
            continue
        out.append({
            "file": str(t.get("filename") or "")[:160],
            "role": t.get("role_group") or t.get("role_primary") or "unknown_mixed",
            "confidence": t.get("role_confidence"),
            "trust_score": t.get("track_trust_score"),
            "trust_tier": t.get("track_trust_tier"),
            "artifact_score": t.get("artifact_score"),
            "cus": t.get("component_utility_score"),
        })
    return out


def _stage_m_count_usable_by_trust(assist: dict[str, Any], minimum: float = 50.0) -> int:
    tracks = assist.get("track_trust_matrix") if isinstance(assist, dict) else []
    count = 0
    if isinstance(tracks, list):
        for t in tracks:
            if not isinstance(t, dict):
                continue
            try:
                if float(t.get("track_trust_score") or 0.0) >= minimum:
                    count += 1
            except Exception:
                continue
    return count


def _build_stage_m_stem_premaster_report(assist: dict[str, Any] | None, *, stem_zip_provided: bool) -> dict[str, Any]:
    """v62.2 Stage M gate for true stem-aware pre-master mixdown.

    The legacy residual/stem-assisted premaster criteria are completely removed
    from Stage M.  This report only trusts the v62.2 stem mixdown builder, which
    performs stem role/reliability analysis, conservative per-role processing,
    reference-constraint validation, and writes a new pre-master mixdown.
    Reconstruction score is diagnostic only and never an absolute-loudness hard
    gate.
    """
    enabled = (not _env_on("BUSY_STAGE_M_REPLACED_BY_STEM_EVIDENCE_ASSIST", "1")) and _env_on("BUSY_STAGE_M_STEM_PREMASTER", "0")
    assist = assist if isinstance(assist, dict) else {}
    builder = assist.get("stage_m_mixdown_builder") if isinstance(assist.get("stage_m_mixdown_builder"), dict) else {}
    stage_m_mixdown = assist.get("stage_m_mixdown_candidate") if isinstance(assist.get("stage_m_mixdown_candidate"), dict) else {}
    if not stage_m_mixdown and isinstance(builder.get("mixdown_candidate"), dict):
        stage_m_mixdown = builder.get("mixdown_candidate") or {}
    builder_validation = builder.get("validation") if isinstance(builder.get("validation"), dict) else {}
    recon = assist.get("global_reconstruction") if isinstance(assist.get("global_reconstruction"), dict) else {}
    summary = assist.get("analysis_summary") if isinstance(assist.get("analysis_summary"), dict) else {}
    role_analysis = builder.get("stem_role_analysis") if isinstance(builder.get("stem_role_analysis"), list) else []
    stem_count_builder = int(builder.get("stem_count") or len(role_analysis) or 0)
    usable_builder = int(builder.get("audio_contribution_stem_count") or builder.get("usable_stem_count") or 0)
    trust_builder = int(builder.get("trust_usable_stem_count") or 0)
    avg_rel = float(builder.get("avg_reliability_score") or 0.0)
    track_count = stem_count_builder or int(assist.get("track_count_analyzed") or 0)
    usable_track_count = usable_builder or int(summary.get("usable_track_count") or 0) if isinstance(summary, dict) else usable_builder
    trust_usable_count = trust_builder or _stage_m_count_usable_by_trust(assist, minimum=float(os.environ.get("BUSY_STAGE_M_USABLE_TRACK_MIN_TRUST", "50")))
    legacy_candidate_count = int(summary.get("limited_processing_candidate_count") or 0) if isinstance(summary, dict) else 0
    global_score = int(recon.get("global_reconstruction_score") or 0) if isinstance(recon, dict) else 0
    global_grade = str(recon.get("global_reconstruction_grade") or "diagnostic_unavailable") if isinstance(recon, dict) else "diagnostic_unavailable"
    recon_available = bool(recon.get("available")) if isinstance(recon, dict) else False

    stage_m_mixdown_accepted = bool(stage_m_mixdown.get("accepted") or builder.get("accepted"))
    stage_m_mixdown_path = str(stage_m_mixdown.get("output_path") or builder.get("output_path") or "")
    stage_m_candidate_count = 1 if (stage_m_mixdown_accepted or builder.get("attempted")) else 0
    candidate_count = stage_m_candidate_count

    fallback_triggers: list[str] = []
    diagnostics: list[str] = []
    if not enabled:
        fallback_triggers.append("BUSY_STAGE_M_STEM_PREMASTER_disabled")
    if not stem_zip_provided:
        fallback_triggers.append("stem_zip_not_provided")
    if stem_zip_provided and track_count <= 0:
        fallback_triggers.append("no_audio_stems_analyzed")
    if stem_zip_provided and not builder:
        fallback_triggers.append("stage_m_mixdown_builder_missing")
    if stem_zip_provided and builder and not builder.get("attempted"):
        diagnostics.append("stage_m_mixdown_builder_not_attempted")
    if stem_zip_provided and global_score and global_score < int(os.environ.get("BUSY_STAGE_M_DIAGNOSTIC_RECON_WARN_SCORE", "60")):
        diagnostics.append("low_gain_solved_reconstruction_score_diagnostic_only")
    if stem_zip_provided and usable_track_count <= 0 and trust_usable_count <= 0:
        fallback_triggers.append("no_role_reliable_usable_stems")
    if stage_m_mixdown_accepted and not stage_m_mixdown_path:
        fallback_triggers.append("stage_m_mixdown_output_path_missing")
    if builder_validation.get("blockers"):
        diagnostics.extend([f"stage_m_validation_blocker:{x}" for x in builder_validation.get("blockers") if x])

    mode = "M4_full_original_stereo_fallback"
    selected_source = "original_stereo"
    reason = "fallback_to_original_stereo_route"
    use_for_mastering = False
    m1_score = float(os.environ.get("BUSY_STAGE_M_M1_MIN_MIXDOWN_SCORE", "0.74"))
    m2_score = float(os.environ.get("BUSY_STAGE_M_M2_MIN_MIXDOWN_SCORE", "0.58"))
    val_score = float(builder_validation.get("score") or 0.0)

    if not enabled:
        reason = "stage_m_disabled"
    elif not stem_zip_provided:
        reason = "no_stem_zip_original_stereo_route"
    elif stage_m_mixdown_accepted and avg_rel >= float(os.environ.get("BUSY_STAGE_M_M1_MIN_AVG_TRUST", "75")) and trust_usable_count >= int(os.environ.get("BUSY_STAGE_M_M1_MIN_USABLE_STEMS", "3")) and val_score >= m1_score:
        mode = "M1_high_confidence_stem_mixdown"
        selected_source = "stage_m_mixdown_candidate"
        reason = "stage_m_role_reliability_reference_mixdown_accepted_high_confidence"
        use_for_mastering = True
    elif stage_m_mixdown_accepted and avg_rel >= float(os.environ.get("BUSY_STAGE_M_M2_MIN_AVG_TRUST", "50")) and trust_usable_count >= int(os.environ.get("BUSY_STAGE_M_M2_MIN_USABLE_STEMS", "1")) and val_score >= m2_score:
        mode = "M2_conservative_stem_mixdown"
        selected_source = "stage_m_mixdown_candidate"
        reason = "stage_m_role_reliability_reference_mixdown_accepted_conservative"
        use_for_mastering = True
    elif stem_zip_provided and (bool(assist.get("usable")) or usable_track_count > 0 or trust_usable_count > 0 or candidate_count > 0 or builder):
        mode = "M3_diagnostic_only_stem_assisted"
        selected_source = "original_stereo"
        reason = "stem_analysis_available_but_stage_m_mixdown_not_accepted_original_route"
        use_for_mastering = False
    else:
        reason = "stem_role_reliability_insufficient_original_route"

    if not _env_on("BUSY_STAGE_M_USE_FOR_MASTERING", "1") and use_for_mastering:
        fallback_triggers.append("BUSY_STAGE_M_USE_FOR_MASTERING_disabled")
        selected_source = "original_stereo"
        reason = "stage_m_audio_use_disabled_original_route"
        use_for_mastering = False
        mode = "M3_diagnostic_only_stem_assisted" if stem_zip_provided else "M4_full_original_stereo_fallback"

    return {
        "schema_version": "busy_stage_m_removed_stem_evidence_gate_v8_5_3_62_9_7",
        "enabled": enabled,
        "attempted": bool(stem_zip_provided and enabled),
        "stem_zip_provided": bool(stem_zip_provided),
        "mode": mode,
        "selected_source": selected_source,
        "use_premaster_for_mastering": bool(use_for_mastering),
        "reason": reason,
        "fallback_triggers": sorted(set(fallback_triggers)),
        "diagnostics": sorted(set(diagnostics)),
        "global_reconstruction_score": global_score,
        "global_reconstruction_grade": global_grade,
        "reconstruction_available": recon_available,
        "reconstruction_policy": "diagnostic_only_lsq_gain_solved_proxy_not_absolute_loudness_gate",
        "old_stem_sum_gate_removed": True,
        "legacy_premaster_removed": True,
        "track_count_analyzed": track_count,
        "usable_track_count": usable_track_count,
        "trust_usable_track_count": trust_usable_count,
        "limited_processing_candidate_count": legacy_candidate_count,
        "stage_m_mixdown_candidate_count": stage_m_candidate_count,
        "avg_track_trust_score": round(avg_rel or _stage_m_avg_track_trust(assist), 3),
        "stage_m_mixdown_candidate": {
            "accepted": stage_m_mixdown_accepted,
            "output_path": stage_m_mixdown_path,
            "reason": stage_m_mixdown.get("reason") or builder.get("reason"),
            "validation_status": builder_validation.get("status") or stage_m_mixdown.get("validation_status"),
            "score": builder_validation.get("score"),
            "blockers": builder_validation.get("blockers", []),
        },
        "stem_roles_compact": [
            {"file": str(x.get("filename") or "")[:160], "role": x.get("role_primary"), "secondary": x.get("role_secondary"), "confidence": x.get("role_confidence"), "reliability_score": x.get("reliability_score"), "tier": x.get("reliability_tier")}
            for x in role_analysis[:10] if isinstance(x, dict)
        ] or _stage_m_role_compact(assist),
        "stage_m_builder_summary": {
            "schema_version": builder.get("schema_version"),
            "accepted": builder.get("accepted"),
            "reason": builder.get("reason"),
            "avg_reliability_score": builder.get("avg_reliability_score"),
            "usable_stem_count": builder.get("usable_stem_count"),
            "audio_contribution_stem_count": builder.get("audio_contribution_stem_count"),
            "trust_usable_stem_count": builder.get("trust_usable_stem_count"),
            "gain_solved_reconstruction_error_db": builder.get("gain_solved_reconstruction_error_db"),
            "raw_gain_solved_reconstruction_error_db_before_v62_9": builder.get("raw_gain_solved_reconstruction_error_db_before_v62_9"),
            "validation": builder_validation,
            "stem_restoration_gate": builder.get("stem_restoration_gate") if isinstance(builder.get("stem_restoration_gate"), dict) else {},
            "full_audio_contribution_enforcement": builder.get("full_audio_contribution_enforcement") if isinstance(builder.get("full_audio_contribution_enforcement"), dict) else {},
            "premix_feasibility_gate": builder.get("premix_feasibility_gate") if isinstance(builder.get("premix_feasibility_gate"), dict) else {},
            "stage_m_mixing_engine": builder.get("stage_m_mixing_engine") if isinstance(builder.get("stage_m_mixing_engine"), dict) else {},
            "reference_intent_profile": builder.get("reference_intent_profile") if isinstance(builder.get("reference_intent_profile"), dict) else {},
            "processing_plan_count": len(builder.get("stem_processing_plan") or []) if isinstance(builder.get("stem_processing_plan"), list) else 0,
            "cross_stem_action_count": len(builder.get("cross_stem_actions") or []) if isinstance(builder.get("cross_stem_actions"), list) else 0,
            "target_zone_action_count": len(builder.get("target_zone_actions") or []) if isinstance(builder.get("target_zone_actions"), list) else 0,
            "full_role_mix_action_count": len(builder.get("full_role_mix_actions") or []) if isinstance(builder.get("full_role_mix_actions"), list) else 0,
            "role_dsp_primitive_action_count": len(builder.get("role_dsp_primitive_actions") or []) if isinstance(builder.get("role_dsp_primitive_actions"), list) else 0,
            "advanced_dsp_module_action_count": len(builder.get("advanced_dsp_module_actions") or []) if isinstance(builder.get("advanced_dsp_module_actions"), list) else 0,
            "target_zone_resolver": builder.get("target_zone_resolver") if isinstance(builder.get("target_zone_resolver"), dict) else {},
            "full_role_mix_rebuild": builder.get("full_role_mix_rebuild") if isinstance(builder.get("full_role_mix_rebuild"), dict) else {},
            "role_dsp_primitives": builder.get("role_dsp_primitives") if isinstance(builder.get("role_dsp_primitives"), dict) else {},
            "advanced_dsp_modules": builder.get("advanced_dsp_modules") if isinstance(builder.get("advanced_dsp_modules"), dict) else {},
            "professional_mix_reconstruction": builder.get("professional_mix_reconstruction") if isinstance(builder.get("professional_mix_reconstruction"), dict) else {},
            "gemini_stem_artifact_mix_intent_review": builder.get("gemini_stem_artifact_mix_intent_review") if isinstance(builder.get("gemini_stem_artifact_mix_intent_review"), dict) else {},
            "safety_summary": builder.get("safety_summary") if isinstance(builder.get("safety_summary"), dict) else {},
            "role_target_zones": builder.get("role_target_zones") if isinstance(builder.get("role_target_zones"), dict) else {},
            "action_specific_safety_criteria": builder.get("action_specific_safety_criteria") if isinstance(builder.get("action_specific_safety_criteria"), dict) else {},
            "evidence_to_action_count": builder.get("evidence_to_action_count"),
            "evidence_to_action_preview": (builder.get("evidence_to_action_inputs") or [])[:20] if isinstance(builder.get("evidence_to_action_inputs"), list) else [],
            "target_priority_map": builder.get("target_priority_map") if isinstance(builder.get("target_priority_map"), dict) else {},
            "stem_processing_plan_preview": (builder.get("stem_processing_plan") or [])[:12] if isinstance(builder.get("stem_processing_plan"), list) else [],
            "cross_stem_actions_preview": (builder.get("cross_stem_actions") or [])[:20] if isinstance(builder.get("cross_stem_actions"), list) else [],
            "target_zone_actions_preview": (builder.get("target_zone_actions") or [])[:20] if isinstance(builder.get("target_zone_actions"), list) else [],
            "full_role_mix_actions_preview": (builder.get("full_role_mix_actions") or [])[:30] if isinstance(builder.get("full_role_mix_actions"), list) else [],
            "role_dsp_primitive_actions_preview": (builder.get("role_dsp_primitive_actions") or [])[:40] if isinstance(builder.get("role_dsp_primitive_actions"), list) else [],
            "advanced_dsp_module_actions_preview": (builder.get("advanced_dsp_module_actions") or [])[:40] if isinstance(builder.get("advanced_dsp_module_actions"), list) else [],
        },
        "policy": {
            "original_stereo_is_safe_baseline": True,
            "original_stereo_is_intent_reference_not_hard_ceiling": True,
            "build_dont_just_sum": True,
            "filenames_are_untrusted_hints": True,
            "legacy_stem_assisted_premaster_removed": True,
            "legacy_stem_assisted_premaster_not_called": True,
            "old_stem_sum_absolute_loudness_gate_removed": True,
            "global_reconstruction_score_is_diagnostic_only": True,
            "stage_m_audio_use_requires_true_role_reliability_reference_mixdown": False,
            "stage_m_mixdown_removed_replaced_by_stem_evidence_assist": True,
            "diagnostic_only_mode_keeps_original_stereo_audio": True,
        },
    }

def _attach_stage_m_report(features: dict[str, Any], decision: dict[str, Any], assist: dict[str, Any], stage_m: dict[str, Any]) -> None:
    if not isinstance(stage_m, dict):
        return
    if isinstance(features, dict):
        features["stage_m_stem_premaster_builder"] = stage_m
    if isinstance(assist, dict):
        assist["stage_m_stem_premaster_builder"] = stage_m
    if isinstance(decision, dict):
        decision.setdefault("input_assist_modes", {})["stage_m_stem_premaster_builder"] = {
            "enabled": stage_m.get("enabled"),
            "attempted": stage_m.get("attempted"),
            "mode": stage_m.get("mode"),
            "selected_source": stage_m.get("selected_source"),
            "use_premaster_for_mastering": stage_m.get("use_premaster_for_mastering"),
            "reason": stage_m.get("reason"),
            "fallback_triggers": stage_m.get("fallback_triggers", []),
            "diagnostics": stage_m.get("diagnostics", []),
            "global_reconstruction_score": stage_m.get("global_reconstruction_score"),
            "reconstruction_policy": stage_m.get("reconstruction_policy"),
            "avg_track_trust_score": stage_m.get("avg_track_trust_score"),
            "trust_usable_track_count": stage_m.get("trust_usable_track_count"),
            "stage_m_mixdown_candidate": stage_m.get("stage_m_mixdown_candidate", {}),
            "downstream_stem_assist_bypassed": stage_m.get("downstream_stem_assist_bypassed"),
            "downstream_stem_assist_bypass_reason": stage_m.get("downstream_stem_assist_bypass_reason"),
            "downstream_stem_assist_policy": stage_m.get("downstream_stem_assist_policy", {}),
            "stage_m_builder_summary": stage_m.get("stage_m_builder_summary", {}),
            "legacy_premaster_removed": True,
        }
        flags = list(decision.get("processing_flags") or []) if isinstance(decision.get("processing_flags"), list) else []
        for flag in ["stage_m_stem_mixdown_gate_available"]:
            if flag not in flags:
                flags.append(flag)
        if stage_m.get("use_premaster_for_mastering") and "stage_m_mixdown_selected_for_mastering" not in flags:
            flags.append("stage_m_mixdown_selected_for_mastering")
        if stage_m.get("selected_source") == "original_stereo" and "stage_m_original_stereo_fallback" not in flags:
            flags.append("stage_m_original_stereo_fallback")
        decision["processing_flags"] = flags


def _get_stage_m_report_from_context(features: Any = None, decision: Any = None, state: Any = None, report: Any = None) -> dict[str, Any]:
    for src in [features, report, state]:
        if isinstance(src, dict) and isinstance(src.get("stage_m_stem_premaster_builder"), dict):
            return src.get("stage_m_stem_premaster_builder") or {}
        if isinstance(src, dict) and isinstance(src.get("input_analysis"), dict) and isinstance(src["input_analysis"].get("stage_m_stem_premaster_builder"), dict):
            return src["input_analysis"].get("stage_m_stem_premaster_builder") or {}
        if isinstance(src, dict) and isinstance(src.get("pre_limiter_report"), dict) and isinstance(src["pre_limiter_report"].get("stage_m_stem_premaster_builder"), dict):
            return src["pre_limiter_report"].get("stage_m_stem_premaster_builder") or {}
    if isinstance(decision, dict):
        iam = decision.get("input_assist_modes") if isinstance(decision.get("input_assist_modes"), dict) else {}
        if isinstance(iam.get("stage_m_stem_premaster_builder"), dict):
            return iam.get("stage_m_stem_premaster_builder") or {}
    return {}


def _attach_stage_m_to_report(report: dict[str, Any], features: Any = None, decision: Any = None, state: Any = None) -> None:
    if not isinstance(report, dict):
        return
    stage_m = _get_stage_m_report_from_context(features=features, decision=decision, state=state, report=report)
    if isinstance(stage_m, dict) and stage_m:
        report["stage_m_stem_premaster_builder"] = stage_m
        prev_pipeline_version = str(report.get("pipeline_version") or "")
        if prev_pipeline_version:
            report["base_stage_m_pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_62_9_9_4_4_stage_m_removed_stem_evidence_input_relative_damage_scoring_lra_metric_count_handoff_hotfix"
        report["pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_63_2_bamix_commercial_premaster_density"
        report["stage_m_selected_source"] = stage_m.get("selected_source")
        report["stage_m_mode"] = stage_m.get("mode")



def _busy_auto_mixing_requested(args: argparse.Namespace | None = None) -> bool:
    raw = "0"
    if args is not None:
        raw = str(getattr(args, "busy_auto_mixing", "") or "")
    if not raw:
        raw = os.environ.get("BUSY_AUTO_MIXING_REQUESTED", os.environ.get("BUSY_AUTOMIX_REQUESTED", "0"))
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _attach_busy_auto_mixing_report(features: dict[str, Any], decision: dict[str, Any], report: dict[str, Any] | None) -> None:
    if not isinstance(report, dict) or not report:
        return
    if isinstance(features, dict):
        features["busy_auto_mixing"] = report
        features["busy_auto_mixing_summary"] = {
            "requested": report.get("requested"),
            "enabled": report.get("enabled"),
            "available": report.get("available"),
            "status": report.get("status"),
            "selected_mix_recipe": report.get("selected_mix_recipe"),
            "premaster_used_for_mastering": report.get("premaster_used_for_mastering"),
            "mastering_input_source": report.get("mastering_input_source"),
            "candidate_render_count": report.get("candidate_render_count"),
            "correction_loops_executed": report.get("correction_loops_executed"),
            "v6370_body_center_vocal_support": report.get("v6370_body_center_vocal_support", {}),
            "v6380_bass_harmonic_translation": report.get("v6380_bass_harmonic_translation", report.get("bass_harmonic_translation", {})),
            "v6390_side_texture_stereo_cleanliness": report.get("v6390_side_texture_stereo_cleanliness", report.get("side_texture_stereo_cleanliness", {})),
            "side_texture_control": report.get("side_texture_control", {}),
        }
    if isinstance(decision, dict):
        iam = decision.setdefault("input_assist_modes", {})
        iam["busy_auto_mixing"] = {
            "requested": report.get("requested"),
            "enabled": report.get("enabled"),
            "available": report.get("available"),
            "status": report.get("status"),
            "selected_mix_recipe": report.get("selected_mix_recipe"),
            "premaster_used_for_mastering": report.get("premaster_used_for_mastering"),
            "mastering_input_source": report.get("mastering_input_source"),
            "reason": report.get("reason"),
            "policy": report.get("candidate_render_policy", {}),
            "v6370_body_center_vocal_support": report.get("v6370_body_center_vocal_support", {}),
            "v6380_bass_harmonic_translation": report.get("v6380_bass_harmonic_translation", report.get("bass_harmonic_translation", {})),
            "v6390_side_texture_stereo_cleanliness": report.get("v6390_side_texture_stereo_cleanliness", report.get("side_texture_stereo_cleanliness", {})),
            "side_texture_control": report.get("side_texture_control", {}),
        }
        flags = list(decision.get("processing_flags") or []) if isinstance(decision.get("processing_flags"), list) else []
        for flag in ["busy_auto_mixing_requested", "busy_auto_mixing_single_recipe_planner", "v6370_body_center_vocal_support_capability_available", "bass_harmonic_translation_capability_available", "v6380_bass_harmonic_translation_capability_available", "v6390_side_texture_stereo_cleanliness_capability_available"]:
            if flag not in flags:
                flags.append(flag)
        if report.get("premaster_used_for_mastering") and "busy_auto_mixing_premaster_used_for_mastering" not in flags:
            flags.append("busy_auto_mixing_premaster_used_for_mastering")
        if report.get("status") in {"technical_failure", "qc_failed", "exception"} and "busy_auto_mixing_failed_explicitly" not in flags:
            flags.append("busy_auto_mixing_failed_explicitly")
        decision["processing_flags"] = flags


def _attach_busy_auto_mixing_to_report(report: dict[str, Any], features: Any = None, decision: Any = None, state: Any = None) -> None:
    if not isinstance(report, dict):
        return
    bamix = {}
    for src in [features, state, report]:
        if isinstance(src, dict) and isinstance(src.get("busy_auto_mixing"), dict):
            bamix = src.get("busy_auto_mixing") or {}
            break
        if isinstance(src, dict) and isinstance(src.get("input_analysis"), dict) and isinstance(src["input_analysis"].get("busy_auto_mixing"), dict):
            bamix = src["input_analysis"].get("busy_auto_mixing") or {}
            break
    if not bamix and isinstance(decision, dict):
        iam = decision.get("input_assist_modes") if isinstance(decision.get("input_assist_modes"), dict) else {}
        if isinstance(iam.get("busy_auto_mixing"), dict):
            bamix = iam.get("busy_auto_mixing") or {}
    if bamix:
        report["busy_auto_mixing"] = bamix
        report["busy_auto_mixing_selected_recipe"] = bamix.get("selected_mix_recipe")
        report["mastering_input_source"] = bamix.get("mastering_input_source", report.get("mastering_input_source"))
        old_engine = report.get("engine_version")
        if bamix.get("premaster_used_for_mastering"):
            report["base_engine_version"] = old_engine
            report["engine_version"] = "busy_auto_mastering_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
        report["pipeline_version"] = "busy_auto_mastering_pipeline_v8_5_3_63_9_0_side_texture_stereo_cleanliness_complete_dsp"
        if isinstance(bamix.get("v6370_body_center_vocal_support"), dict):
            report["v6370_body_center_vocal_support"] = bamix.get("v6370_body_center_vocal_support")
            report["body_center_vocal_support"] = bamix.get("v6370_body_center_vocal_support")
        if isinstance(bamix.get("v6380_bass_harmonic_translation"), dict):
            report["v6380_bass_harmonic_translation"] = bamix.get("v6380_bass_harmonic_translation")
            report["bass_harmonic_translation"] = bamix.get("v6380_bass_harmonic_translation")
        elif isinstance(bamix.get("bass_harmonic_translation"), dict):
            report["v6380_bass_harmonic_translation"] = bamix.get("bass_harmonic_translation")
            report["bass_harmonic_translation"] = bamix.get("bass_harmonic_translation")
        if isinstance(bamix.get("v6390_side_texture_stereo_cleanliness"), dict):
            report["v6390_side_texture_stereo_cleanliness"] = bamix.get("v6390_side_texture_stereo_cleanliness")
            report["side_texture_stereo_cleanliness"] = bamix.get("v6390_side_texture_stereo_cleanliness")
        elif isinstance(bamix.get("side_texture_stereo_cleanliness"), dict):
            report["v6390_side_texture_stereo_cleanliness"] = bamix.get("side_texture_stereo_cleanliness")
            report["side_texture_stereo_cleanliness"] = bamix.get("side_texture_stereo_cleanliness")
        _v6390_report = report.get("v6390_side_texture_stereo_cleanliness") if isinstance(report.get("v6390_side_texture_stereo_cleanliness"), dict) else {}
        if _v6390_report:
            report["side_texture_control"] = _v6390_report.get("side_texture_control", report.get("side_texture_control", {}))
            report["side_high_hash_fizz_suppressor"] = _v6390_report.get("side_high_hash_fizz_suppressor", report.get("side_high_hash_fizz_suppressor", {}))
            report["stereo_cleanliness_guard"] = _v6390_report.get("stereo_cleanliness_guard", report.get("stereo_cleanliness_guard", {}))
            report["ambience_collapse_detection"] = _v6390_report.get("ambience_collapse_detection", report.get("ambience_collapse_detection", {}))
            report["mono_fold_down_rollback"] = _v6390_report.get("mono_fold_down_rollback", report.get("mono_fold_down_rollback", {}))


def _apply_busy_auto_mixing_if_requested(
    *,
    y: np.ndarray,
    sr: int,
    features: dict[str, Any],
    decision: dict[str, Any],
    preflight_report: dict[str, Any] | None,
    local_input: Path,
    local_stem_zip: Path | None,
    args: argparse.Namespace,
    log_fp,
    work_dir: Path,
) -> tuple[np.ndarray, int, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    requested = _busy_auto_mixing_requested(args)
    if not requested:
        _clear_bamix_runtime_handoff_env()
        return y, sr, features, decision, preflight_report
    if local_stem_zip is None or not Path(local_stem_zip).exists():
        bamix = {
            "schema_version": "busy_auto_mixing_v8_5_3_63_2_commercial_premaster_density",
            "requested": True,
            "enabled": True,
            "available": False,
            "status": "technical_failure",
            "reason": "stem_zip_required_for_busy_auto_mixing",
            "failure_reasons": ["stem_zip_missing"],
            "premaster_used_for_mastering": False,
            "mastering_input_source": "original_stereo_not_used_because_bamix_failed_explicitly",
            "policy": {"no_silent_original_stereo_fallback": True},
        }
        _clear_bamix_runtime_handoff_env()
        _attach_busy_auto_mixing_report(features, decision, bamix)
        _log(log_fp, "busy_auto_mixing_missing_stem_zip", requested=True)
        if _env_on("BUSY_AUTOMIX_FAIL_ON_TECHNICAL_FAILURE", os.environ.get("BUSY_AUTO_MIXING_FAIL_ON_TECHNICAL_FAILURE", "1")):
            raise RuntimeError("BUSY_MIX_FAILURE: Busy Auto Mixing was requested but no stem ZIP was available.")
        return y, sr, features, decision, preflight_report

    _log(log_fp, "stage_start_busy_auto_mixing", stem_zip=str(local_stem_zip), original=str(local_input), ai_model=os.environ.get("BUSY_AUTOMIX_AI_MODEL", "gpt-5.5"))
    premaster_path, bamix_report = build_busy_auto_mixing_premaster(
        stem_zip_path=local_stem_zip,
        original_stereo_path=local_input,
        work_dir=work_dir,
        original_features=features,
        original_decision=decision,
        log_callback=lambda event, **fields: _log(log_fp, event, **fields),
        private_docs_zip_path=args.private_zip,
    )
    _attach_busy_auto_mixing_report(features, decision, bamix_report)
    if not premaster_path or not isinstance(bamix_report, dict) or not bamix_report.get("premaster_used_for_mastering"):
        qc = (bamix_report or {}).get("premaster_qc") if isinstance(bamix_report, dict) else {}
        if not isinstance(qc, dict):
            qc = {}
        failure_digest = {
            "status": (bamix_report or {}).get("status") if isinstance(bamix_report, dict) else None,
            "reason": (bamix_report or {}).get("reason") if isinstance(bamix_report, dict) else None,
            "failure_reasons": (bamix_report or {}).get("failure_reasons") if isinstance(bamix_report, dict) else None,
            "schema_version": (bamix_report or {}).get("schema_version") if isinstance(bamix_report, dict) else None,
            "premaster_qc_hard_fail": qc.get("hard_fail"),
            "premaster_qc_hard_warnings": qc.get("hard_warnings"),
            "premaster_qc_warnings": qc.get("warnings"),
            "premaster_qc_lufs": qc.get("integrated_lufs"),
            "premaster_qc_true_peak_dbtp": qc.get("true_peak_dbtp"),
            "premaster_qc_crest_factor_db": qc.get("crest_factor_db"),
            "premaster_qc_phase_correlation": qc.get("phase_correlation"),
            "v63361_qc_salvage": (bamix_report or {}).get("v63361_qc_salvage") if isinstance(bamix_report, dict) else None,
            "render_attempt_count": len((bamix_report or {}).get("render_attempts") or []) if isinstance(bamix_report, dict) else None,
            "qc_attempt_count": len((bamix_report or {}).get("premaster_qc_attempts") or []) if isinstance(bamix_report, dict) else None,
        }
        try:
            features["busy_auto_mixing_failure_digest"] = failure_digest
            decision.setdefault("input_assist_modes", {}).setdefault("busy_auto_mixing", {})["failure_digest"] = failure_digest
        except Exception:
            pass
        _log(log_fp, "busy_auto_mixing_failed", **failure_digest)
        _clear_bamix_runtime_handoff_env()
        if _env_on("BUSY_AUTOMIX_FAIL_ON_MIX_FAILURE", os.environ.get("BUSY_AUTO_MIXING_FAIL_ON_MIX_FAILURE", "1")):
            try:
                detail = json.dumps(failure_digest, ensure_ascii=False, default=_json_default)[:1800]
            except Exception:
                detail = str(failure_digest)[:1800]
            raise RuntimeError("BUSY_MIX_FAILURE: Busy Auto Mixing failed and original-stereo fallback is disabled for explicit BAMix requests. detail=" + detail)
        return y, sr, features, decision, preflight_report

    try:
        y = None  # type: ignore[assignment]
        gc.collect()
    except Exception:
        pass
    bamix_report["memory_handoff_barrier_before_mastering"] = _bamix_memory_handoff_barrier(log_fp, phase="after_bamix_before_premaster_reanalysis")
    _log(log_fp, "stage_start_busy_auto_mixing_premaster_reanalysis", path=str(premaster_path), bytes=Path(premaster_path).stat().st_size)
    y2, sr2, features2, decision2, preflight2 = _build_features_and_decision(
        Path(premaster_path),
        args.private_zip,
        args.mode,
        log_fp,
        args.user_style_hint,
        args.private_rulepack,
        stem_zip_path=None,
        work_dir=work_dir,
    )
    bamix_report["premaster_reanalyzed_for_mastering"] = True
    bamix_report["effective_input_path"] = str(premaster_path)
    bamix_report["mastering_input_source"] = "busy_auto_mixing_premaster"
    bamix_report["original_stereo_used_as"] = "reference_anchor"
    features2["effective_input_mode"] = "busy_auto_mixing_premaster"
    features2["busy_auto_mixing_premaster_used_for_mastering"] = True
    # Avoid applying separated-stem assist twice after the premaster is already built from stems.
    features2["multi_component_assist_summary"] = {
        "provided": True,
        "downstream_stem_assist_bypassed": True,
        "downstream_bypass_reason": "busy_auto_mixing_premaster_selected_as_effective_input",
        "lgcca_attached_to_mastering_decision": False,
        "ssrw_attached_to_mastering_decision": False,
    }
    features2["stem_evidence_assist"] = {
        "enabled": False,
        "bypassed": True,
        "bypass_reason": "busy_auto_mixing_premaster_selected_as_effective_input",
    }
    _attach_busy_auto_mixing_report(features2, decision2, bamix_report)
    _apply_bamix_mastering_handoff_policy(features2, decision2, bamix_report, log_fp)
    _attach_busy_auto_mixing_report(features2, decision2, bamix_report)
    decision2.setdefault("input_assist_modes", {})["effective_audio_source"] = "busy_auto_mixing_premaster"
    flags = list(decision2.get("processing_flags") or []) if isinstance(decision2.get("processing_flags"), list) else []
    for flag in [
        "busy_auto_mixing_requested",
        "busy_auto_mixing_premaster_generated",
        "busy_auto_mixing_premaster_used_for_mastering",
        "busy_auto_mixing_downstream_stem_assist_bypassed",
        "busy_auto_mixing_no_multi_candidate_full_render",
    ]:
        if flag not in flags:
            flags.append(flag)
    decision2["processing_flags"] = flags
    _log(log_fp, "stage_done_busy_auto_mixing_premaster_reanalysis", sr=sr2, shape=list(y2.shape), effective_input="busy_auto_mixing_premaster", recipe=bamix_report.get("selected_mix_recipe"))
    return y2, sr2, features2, decision2, preflight2

def _select_stage_m_effective_input(
    *,
    y: np.ndarray,
    sr: int,
    features: dict[str, Any],
    decision: dict[str, Any],
    preflight_report: dict[str, Any] | None,
    assist: dict[str, Any],
    args: argparse.Namespace,
    log_fp,
) -> tuple[np.ndarray, int, dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Stage M v62.2 effective-input selector.

    Despite the historical function name, this no longer uses or references the
    legacy residual/stem-assisted premaster.  The only audio replacement path is
    a true Stage M reference-constrained stem mixdown candidate generated by
    audio_engine.stage_m_mixdown_builder.  If that candidate is absent, rejected,
    missing, or disabled by env, the original stereo remains the effective input.
    """
    stem_zip_provided = bool(isinstance(assist, dict) and assist.get("provided"))
    stage_m = (assist.get("stage_m_stem_premaster_builder") if isinstance(assist, dict) and isinstance(assist.get("stage_m_stem_premaster_builder"), dict) else None)
    if stage_m is None:
        stage_m = _build_stage_m_stem_premaster_report(assist, stem_zip_provided=stem_zip_provided)
        _attach_stage_m_report(features, decision, assist, stage_m)
    _log(log_fp, "stage_m_stem_mixdown_gate", mode=stage_m.get("mode"), selected_source=stage_m.get("selected_source"), use_premaster=stage_m.get("use_premaster_for_mastering"), reason=stage_m.get("reason"), fallback_triggers=stage_m.get("fallback_triggers"), diagnostics=stage_m.get("diagnostics"))

    mix_cand = stage_m.get("stage_m_mixdown_candidate") if isinstance(stage_m.get("stage_m_mixdown_candidate"), dict) else {}
    if not stage_m.get("use_premaster_for_mastering"):
        return y, sr, features, decision, preflight_report
    if not _env_on("BUSY_STAGE_M_USE_FOR_MASTERING", "1"):
        stage_m["use_premaster_for_mastering"] = False
        stage_m["used_for_mastering"] = False
        stage_m["selected_source"] = "original_stereo"
        stage_m["reason"] = "BUSY_STAGE_M_USE_FOR_MASTERING_disabled"
        stage_m.setdefault("fallback_triggers", []).append("BUSY_STAGE_M_USE_FOR_MASTERING_disabled")
        _attach_stage_m_report(features, decision, assist, stage_m)
        return y, sr, features, decision, preflight_report
    mix_path = Path(str(mix_cand.get("output_path") or ""))
    if not mix_cand.get("accepted") or not str(mix_path):
        stage_m["use_premaster_for_mastering"] = False
        stage_m["used_for_mastering"] = False
        stage_m["selected_source"] = "original_stereo"
        stage_m["reason"] = "stage_m_mixdown_not_accepted_original_route"
        stage_m.setdefault("fallback_triggers", []).append("stage_m_mixdown_not_accepted")
        _attach_stage_m_report(features, decision, assist, stage_m)
        return y, sr, features, decision, preflight_report
    if not mix_path.exists():
        _log(log_fp, "stage_m_mixdown_missing_after_accept", path=str(mix_path))
        stage_m["use_premaster_for_mastering"] = False
        stage_m["used_for_mastering"] = False
        stage_m["selected_source"] = "original_stereo"
        stage_m["reason"] = "stage_m_mixdown_output_path_missing"
        stage_m.setdefault("fallback_triggers", []).append("stage_m_mixdown_output_path_missing")
        _attach_stage_m_report(features, decision, assist, stage_m)
        return y, sr, features, decision, preflight_report

    _log(log_fp, "stage_start_stage_m_mixdown_reanalysis", path=str(mix_path), bytes=mix_path.stat().st_size)
    try:
        y = None  # type: ignore[assignment]
        gc.collect()
    except Exception:
        pass
    y2, sr2, features2, decision2, preflight2 = _build_features_and_decision(
        mix_path,
        args.private_zip,
        args.mode,
        log_fp,
        args.user_style_hint,
        args.private_rulepack,
    )
    stage_m["used_for_mastering"] = True
    stage_m["effective_input_path"] = str(mix_path)
    stage_m["effective_input_mode"] = "stage_m_reference_constrained_stem_mixdown"
    stage_m["downstream_stem_assist_bypassed"] = True
    stage_m["downstream_stem_assist_bypass_reason"] = "stage_m_mixdown_is_effective_mastering_input"
    stage_m["downstream_stem_assist_policy"] = {
        "skip_when_stage_m_mixdown_selected": True,
        "skip_when_no_stem_zip": True,
        "keep_when_original_stereo_selected_with_stems": True,
        "reason": "avoid applying stem-derived correction again after stems were already used to build the pre-master mixdown",
    }
    assist["stage_m_stem_premaster_builder"] = stage_m
    assist["downstream_multicomponent_assist_bypassed"] = True
    assist["downstream_bypass_reason"] = "stage_m_mixdown_selected_as_effective_input"

    # Important v62.2.2 routing rule:
    # The selected Stage M mixdown is already the stem-derived pre-master.  Do
    # not attach LG-CCA/SSRW/multicomponent control maps to the new mastering
    # decision, otherwise mastering would use the same stems twice.
    features2["multi_component_assist_summary"] = {
        "provided": assist.get("provided"),
        "downstream_stem_assist_bypassed": True,
        "downstream_bypass_reason": "stage_m_mixdown_selected_as_effective_input",
        "lgcca_attached_to_mastering_decision": False,
        "ssrw_attached_to_mastering_decision": False,
    }
    features2["low_grs_clean_component_assist"] = {
        "enabled": False,
        "active": False,
        "bypassed": True,
        "bypass_reason": "stage_m_mixdown_selected_as_effective_input",
    }
    features2["stem_sum_residue_witness"] = {
        "enabled": False,
        "bypassed": True,
        "bypass_reason": "stage_m_mixdown_selected_as_effective_input",
    }
    _attach_stage_m_report(features2, decision2, assist, stage_m)
    features2["effective_input_mode"] = "stage_m_reference_constrained_stem_mixdown"
    features2["stage_m_stem_mixdown_used_for_mastering"] = True
    decision2.setdefault("input_assist_modes", {})["effective_audio_source"] = "stage_m_reference_constrained_stem_mixdown"
    flags = list(decision2.get("processing_flags") or []) if isinstance(decision2.get("processing_flags"), list) else []
    for flag in [
        "stage_m_stem_mixdown_generated",
        "stage_m_stem_mixdown_used_for_mastering",
        "stage_m_downstream_stem_assist_bypassed",
        "stage_m_lgcca_ssrw_bypassed_for_effective_mixdown",
    ]:
        if flag not in flags:
            flags.append(flag)
    decision2["processing_flags"] = flags
    _log(log_fp, "stage_done_stage_m_mixdown_reanalysis", sr=sr2, shape=list(y2.shape), effective_input="stage_m_reference_constrained_stem_mixdown")
    return y2, sr2, features2, decision2, preflight2

def run_single_stage(args: argparse.Namespace, log_fp) -> int:
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    local_input = Path(args.input) if args.input else work_dir / "input_from_supabase.wav"
    local_output = Path(args.output) if args.output else work_dir / "output_to_supabase.wav"
    if args.supabase_input:
        _log(log_fp, "stage_start_supabase_download", path=args.supabase_input)
        storage = SupabaseStorage.from_env()
        storage.download_file(args.supabase_input, str(local_input))
        _log(log_fp, "stage_done_supabase_download", local=str(local_input), bytes=local_input.stat().st_size)
    local_stem_zip = _download_optional_multicomponent_zip(args, work_dir, log_fp)
    premaster_output_path = None  # v62.2: legacy residual stem-assisted premaster is not generated
    # v62.4.1: Stage M must not render before the original stereo analysis/decision package exists.
    # Parallel stem assist is intentionally bypassed for Stage M ordering correctness.
    y, sr, features, decision, _preflight = _build_features_and_decision(local_input, args.private_zip, args.mode, log_fp, args.user_style_hint, args.private_rulepack, stem_zip_path=local_stem_zip, work_dir=work_dir)
    # v8.5.3.62.9.7: Stage M mixdown is removed.  Stem ZIP evidence has
    # already been consumed before Suno-artifact analysis and GPT decision.
    # Keep an explicit stage_m report for telemetry, but never replace y.
    assist = features.get("multi_component_assist") if isinstance(features.get("multi_component_assist"), dict) else {
        "provided": bool(local_stem_zip),
        "enabled": _env_on("BUSY_STEM_EVIDENCE_ASSIST", "1"),
        "usable": False,
        "recommended_mode": "stem_evidence_assist_unavailable",
        "separated_tracks_used_for_audio_render": False,
    }
    stage_m_removed = _build_stage_m_stem_premaster_report(assist, stem_zip_provided=bool(local_stem_zip))
    _attach_stage_m_report(features, decision, assist, stage_m_removed)
    _log(log_fp, "stage_m_removed_original_stereo_route", stem_zip_provided=bool(local_stem_zip), reason=stage_m_removed.get("reason"), mode=stage_m_removed.get("mode"), selected_source=stage_m_removed.get("selected_source"))
    y, sr, features, decision, _preflight = _apply_busy_auto_mixing_if_requested(y=y, sr=sr, features=features, decision=decision, preflight_report=_preflight, local_input=local_input, local_stem_zip=local_stem_zip, args=args, log_fp=log_fp, work_dir=work_dir)
    features, decision, _v6360_fw = _attach_v6360_research_capability_framework_if_missing(
        features,
        decision,
        stage="single_stage_after_bamix_before_master_audio",
        log_fp=log_fp,
    )
    _log(log_fp, "stage_start_master_audio")
    mastered, report = master_audio(y, sr, decision, mode=args.mode, pre_analysis=features)
    _attach_stage_m_to_report(report, features=features, decision=decision)
    _attach_busy_auto_mixing_to_report(report, features=features, decision=decision)
    report = _v6314_4_finalize_floor_tolerance_and_handoff_markers(report, decision)
    report = _v6362_8_finalize_report_authority(report, decision)
    report = attach_research_capability_to_report(report, features=features, decision=decision, stage="single_stage_final_report")
    # v63.6.2.11: Research Capability Framework attachment is report-only, but
    # it stamps framework engine/schema labels.  Re-apply DML report authority
    # after that attachment so the delivered report source/schema remains final.
    report = _v6362_8_finalize_report_authority(report, decision)
    y = None
    gc.collect()
    _log(log_fp, "stage_done_master_audio", output_shape=list(mastered.shape), output_lufs=report.get("output_analysis", {}).get("integrated_lufs"), output_true_peak=report.get("output_analysis", {}).get("approx_true_peak_dbfs"))
    _log(log_fp, "stage_start_write_wav")
    write_wav_pcm24(str(local_output), mastered, sr=48_000)
    mastered = None
    gc.collect()
    _log(log_fp, "stage_done_write_wav", output=str(local_output))
    if args.supabase_output:
        _log(log_fp, "stage_start_supabase_upload", path=args.supabase_output)
        storage = SupabaseStorage.from_env()
        storage.upload_file(args.supabase_output, str(local_output), content_type="audio/wav")
        _log(log_fp, "stage_done_supabase_upload", path=args.supabase_output, bytes=local_output.stat().st_size)
    report = attach_finishing_intelligence_to_report(report, state={}, decision=decision, output_path=local_output)
    report = attach_research_capability_to_report(report, features=features, decision=decision, stage="single_stage_v6400_final_report")
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    debug_brief = _write_debug_brief_if_requested(args, report, {}, log_fp)
    if isinstance(debug_brief, dict) and debug_brief.get("active"):
        report["debug_brief"] = debug_brief
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _log(log_fp, "worker_done", elapsed_sec=round(time.time() - t0, 3), two_stage=False)
    return 0


def run_stage_analyze(args: argparse.Namespace, log_fp) -> int:
    """Stage A-1: load input, preflight-clean, analyze, decide, then checkpoint clean float32 audio.

    This worker intentionally exits before the pre-limiter render.  The next worker
    reloads both audio and state from Supabase so NumPy/SciPy/GPT/private-rule memory
    from A-1 cannot stack with A-2 DSP memory.
    """
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    local_input = Path(args.input) if args.input else work_dir / "input_from_supabase.wav"
    local_preprocessed = Path(args.preprocessed or (work_dir / "preprocessed_float32.wav"))
    if args.supabase_input:
        _log(log_fp, "stage_start_supabase_download", path=args.supabase_input)
        storage = SupabaseStorage.from_env()
        storage.download_file(args.supabase_input, str(local_input))
        _log(log_fp, "stage_done_supabase_download", local=str(local_input), bytes=local_input.stat().st_size)

    local_stem_zip = _download_optional_multicomponent_zip(args, work_dir, log_fp)
    premaster_output_path = None  # v62.2: legacy residual stem-assisted premaster is not generated
    # v62.4.1: Stage M must run after original stereo analysis/decision so it can reuse
    # Essentia/Realtime/evidence/profile/mix-conditioning intent rather than guessing early.
    y, sr, features, decision, preflight_report = _build_features_and_decision(local_input, args.private_zip, args.mode, log_fp, args.user_style_hint, args.private_rulepack, stem_zip_path=local_stem_zip, work_dir=work_dir)
    # v8.5.3.62.9.7: Stage M mixdown is removed.  Stem Evidence Assist ran
    # inside _build_features_and_decision before artifact analysis/decision.
    assist = features.get("multi_component_assist") if isinstance(features.get("multi_component_assist"), dict) else {
        "provided": bool(local_stem_zip),
        "enabled": _env_on("BUSY_STEM_EVIDENCE_ASSIST", "1"),
        "usable": False,
        "recommended_mode": "stem_evidence_assist_unavailable",
        "separated_tracks_used_for_audio_render": False,
    }
    stage_m_removed = _build_stage_m_stem_premaster_report(assist, stem_zip_provided=bool(local_stem_zip))
    _attach_stage_m_report(features, decision, assist, stage_m_removed)
    _log(log_fp, "stage_m_removed_original_stereo_route", stem_zip_provided=bool(local_stem_zip), reason=stage_m_removed.get("reason"), mode=stage_m_removed.get("mode"), selected_source=stage_m_removed.get("selected_source"))
    y, sr, features, decision, preflight_report = _apply_busy_auto_mixing_if_requested(y=y, sr=sr, features=features, decision=decision, preflight_report=preflight_report, local_input=local_input, local_stem_zip=local_stem_zip, args=args, log_fp=log_fp, work_dir=work_dir)
    features, decision, _v6360_fw = _attach_v6360_research_capability_framework_if_missing(
        features,
        decision,
        stage="A1_after_bamix_before_state",
        log_fp=log_fp,
    )

    local_preprocessed.parent.mkdir(parents=True, exist_ok=True)
    _log(log_fp, "stage_start_write_preprocessed_float32", path=str(local_preprocessed))
    sf.write(str(local_preprocessed), np.asarray(y, dtype=np.float32), sr, format="WAV", subtype="FLOAT")
    y = None
    gc.collect()
    _log(log_fp, "stage_done_write_preprocessed_float32", path=str(local_preprocessed), bytes=local_preprocessed.stat().st_size)

    if args.supabase_preprocessed:
        _log(log_fp, "stage_start_preprocessed_upload", path=args.supabase_preprocessed)
        storage = SupabaseStorage.from_env()
        storage.upload_file(args.supabase_preprocessed, str(local_preprocessed), content_type="audio/wav")
        _log(log_fp, "stage_done_preprocessed_upload", path=args.supabase_preprocessed, bytes=local_preprocessed.stat().st_size)

    state = {
        "state_schema": "busy_stage_a1_v2",
        "stage": "A-1",
        "stage_a1_completed_at_utc": _now(),
        "job_id": _stage_job_id_from_path(args.supabase_input or args.supabase_preprocessed or args.supabase_stage_state),
        "source_input_supabase_path": args.supabase_input,
        "mode": args.mode,
        "mastering_intensity": _effective_mastering_intensity(decision, getattr(args, "mastering_intensity", "auto")),
        "auto_intensity_policy": decision.get("auto_intensity_policy", {}),
        "sr": sr,
        "decision": decision,
        "decision_digest": _decision_profile_digest(decision),
        "input_analysis": features,
        "analysis_sources": _analysis_sources_snapshot(features, decision),
        "preflight_report": preflight_report,
        "pre_clean_plan": features.get("pre_clean_plan", {}),
        "post_pre_clean_analysis": features.get("post_pre_clean_analysis", {}),
        "mix_conditioning_plan": features.get("mix_conditioning_plan", {}),
        "mix_conditioning_analysis": features.get("mix_conditioning_analysis", {}),
        "mix_conditioning_validation": features.get("mix_conditioning_validation", {}),
        "mix_conditioning_actions": features.get("mix_conditioning_actions", []),
        "mix_conditioning_feature_deltas": features.get("mix_conditioning_feature_deltas", {}),
        "loudness_feasibility_predictor": features.get("loudness_feasibility_predictor", {}),
        "perceptual_quality_score_baseline": features.get("perceptual_quality_score_baseline", {}),
        "multi_criteria_stress_window_requirements": features.get("multi_criteria_stress_window_requirements", {}),
        "multi_component_assist": features.get("multi_component_assist", {}),
        "multi_component_assist_summary": features.get("multi_component_assist_summary", {}),
        "stem_evidence_assist": features.get("stem_evidence_assist", {}),
        "stage_m_stem_premaster_builder": features.get("stage_m_stem_premaster_builder", {}),
        "busy_auto_mixing": features.get("busy_auto_mixing", {}),
        "busy_auto_mixing_summary": features.get("busy_auto_mixing_summary", {}),
        "v6370_body_center_vocal_support": (features.get("busy_auto_mixing", {}) or {}).get("v6370_body_center_vocal_support", {}) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "v6380_bass_harmonic_translation": (features.get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", (features.get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", {})) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "v6390_side_texture_stereo_cleanliness": (features.get("busy_auto_mixing", {}) or {}).get("v6390_side_texture_stereo_cleanliness", (features.get("busy_auto_mixing", {}) or {}).get("side_texture_stereo_cleanliness", {})) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "side_texture_control": (features.get("busy_auto_mixing", {}) or {}).get("side_texture_control", {}) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "low_grs_clean_component_assist": features.get("low_grs_clean_component_assist", {}),
        "blocker_remediation_plan": features.get("blocker_remediation_plan", {}),
        "post_remediation_predicted_metrics": features.get("post_remediation_predicted_metrics", {}),
        "optimized_pipeline_architecture": features.get("optimized_pipeline_architecture", {}),
        "render_budget_policy": decision.get("render_budget_policy", {}),
        "research_capability_framework": features.get("research_capability_framework", decision.get("research_capability_framework", {})),
        "research_capability_summary": compact_research_capability_summary(features.get("research_capability_framework", decision.get("research_capability_framework", {}))),
        "remaining_capability_matrix": features.get("remaining_capability_matrix", decision.get("remaining_capability_matrix", [])),
        "capability_router": features.get("capability_router", decision.get("capability_router", {})),
        "capability_contracts": features.get("capability_contracts", decision.get("capability_contracts", {})),
        "capability_conflict_policy": features.get("capability_conflict_policy", decision.get("capability_conflict_policy", {})),
        "preprocessed_format": "float32_wav",
        "preprocessed_supabase_path": args.supabase_preprocessed,
        "intermediate_supabase_path": args.supabase_intermediate,
    }
    if args.stage_state:
        _write_json_file(args.stage_state, state)
        _log(log_fp, "stage_done_write_stage_state", path=args.stage_state, has_analysis_sources=bool(state.get("analysis_sources")), has_realtime_audio_judge=bool(((state.get("input_analysis") or {}).get("realtime_audio_judge") if isinstance(state.get("input_analysis"), dict) else None)), has_essentia_tensorflow=bool((((state.get("input_analysis") or {}).get("essentia_evidence") or {}).get("tensorflow_tags") if isinstance((state.get("input_analysis") or {}).get("essentia_evidence"), dict) else None)) if isinstance(state.get("input_analysis"), dict) else False)
        _upload_stage_state_if_requested(args, log_fp)

    _analysis_engine_version = "busy_auto_mastering_v8_5_3_63_1_4_2_busy_auto_mixing_analyze_checkpoint" if isinstance(features.get("busy_auto_mixing"), dict) and features.get("busy_auto_mixing") else "busy_auto_mastering_v7_6_checkpointed_analysis"
    analysis_report = {
        "engine_version": _analysis_engine_version,
        "stage": "analyze_checkpoint",
        "input_analysis": features,
        "decision_summary": {
            "detected_style": decision.get("detected_style") or decision.get("genre_id"),
            "confidence": decision.get("style_confidence"),
            "reference_v7": decision.get("reference_v7", {}),
        },
        "preflight_report": preflight_report,
        "analysis_sources": _analysis_sources_snapshot(features, decision),
        "multi_component_assist": features.get("multi_component_assist", {}),
        "multi_component_assist_summary": features.get("multi_component_assist_summary", {}),
        "stem_evidence_assist": features.get("stem_evidence_assist", {}),
        "stage_m_stem_premaster_builder": features.get("stage_m_stem_premaster_builder", {}),
        "busy_auto_mixing": features.get("busy_auto_mixing", {}),
        "busy_auto_mixing_summary": features.get("busy_auto_mixing_summary", {}),
        "v6370_body_center_vocal_support": (features.get("busy_auto_mixing", {}) or {}).get("v6370_body_center_vocal_support", {}) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "v6380_bass_harmonic_translation": (features.get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", (features.get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", {})) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "v6390_side_texture_stereo_cleanliness": (features.get("busy_auto_mixing", {}) or {}).get("v6390_side_texture_stereo_cleanliness", (features.get("busy_auto_mixing", {}) or {}).get("side_texture_stereo_cleanliness", {})) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "side_texture_control": (features.get("busy_auto_mixing", {}) or {}).get("side_texture_control", {}) if isinstance(features.get("busy_auto_mixing"), dict) else {},
        "research_capability_framework": features.get("research_capability_framework", decision.get("research_capability_framework", {})),
        "research_capability_summary": compact_research_capability_summary(features.get("research_capability_framework", decision.get("research_capability_framework", {}))),
        "remaining_capability_matrix": features.get("remaining_capability_matrix", decision.get("remaining_capability_matrix", [])),
        "capability_router": features.get("capability_router", decision.get("capability_router", {})),
        "capability_contracts": features.get("capability_contracts", decision.get("capability_contracts", {})),
        "capability_conflict_policy": features.get("capability_conflict_policy", decision.get("capability_conflict_policy", {})),
        "preprocessed_supabase_path": args.supabase_preprocessed,
    }
    analysis_report = attach_research_capability_to_report(analysis_report, features=features, decision=decision, stage="A1_analysis_checkpoint_report")
    Path(args.report).write_text(json.dumps(analysis_report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _log(log_fp, "worker_stage_analyze_done", elapsed_sec=round(time.time() - t0, 3))
    return 0



def _load_a1_state_and_preprocessed(args: argparse.Namespace, log_fp) -> tuple[np.ndarray, int, dict[str, Any], dict[str, Any], dict[str, Any]]:
    work_dir = Path(args.report).resolve().parent
    local_preprocessed = Path(args.preprocessed or (work_dir / "preprocessed_float32.wav"))
    if args.supabase_preprocessed:
        _download_stage_state_if_requested(args, log_fp)
    if not args.stage_state or not Path(args.stage_state).exists():
        raise RuntimeError("--stage-state is required and must exist for Stage A-2 split workers")
    state_in = json.loads(Path(args.stage_state).read_text(encoding="utf-8"))
    decision = _canonicalize_decision_for_stage(state_in.get("decision", {}), stage="A-2_state_loaded")
    state_in["decision"] = decision
    state_in["decision_digest"] = _decision_profile_digest(decision)
    features = state_in.get("input_analysis", {})
    _log(log_fp, "stage_state_decision_loaded", **_decision_profile_digest(decision), state_job_id=state_in.get("job_id"), path=args.stage_state)
    _apply_intensity_from_stage_state(state_in, args, log_fp, phase="A_state_load")
    if not decision.get("selected_profile"):
        raise RuntimeError("Stage state is missing selected_profile; refusing to continue with unknown mastering profile")
    if args.supabase_preprocessed:
        _log(log_fp, "stage_start_preprocessed_download", path=args.supabase_preprocessed)
        storage = SupabaseStorage.from_env()
        storage.download_file(args.supabase_preprocessed, str(local_preprocessed))
        _log(log_fp, "stage_done_preprocessed_download", local=str(local_preprocessed), bytes=local_preprocessed.stat().st_size)
    _log(log_fp, "stage_start_load_preprocessed_float32")
    y, sr = load_audio_file(str(local_preprocessed), target_sr=48_000, dtype="float32")
    _log(log_fp, "stage_done_load_preprocessed_float32", sr=sr, shape=list(y.shape), dtype=str(y.dtype), seconds=round(y.shape[0] / sr, 3))
    return y, sr, features, decision, state_in


def run_stage_pre_prepare(args: argparse.Namespace, log_fp) -> int:
    """Stage A-2a: working-level staging + governed decision checkpoint."""
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    local_a2a = Path(args.a2a or (work_dir / "a2a_prepared_float32.wav"))
    y, sr, features, decision, state = _load_a1_state_and_preprocessed(args, log_fp)

    _log(log_fp, "stage_start_pre_limiter_prepare")
    staged, render_decision, prepare_report = prepare_pre_limiter_stage(y, sr, decision, mode=args.mode, pre_analysis=features)
    render_decision = _canonicalize_decision_for_stage(render_decision, stage="A-2a_render_decision")
    y = None
    gc.collect()
    _log(log_fp, "stage_done_pre_limiter_prepare", output_shape=list(staged.shape), stage_gain_db=prepare_report.get("working_stage_gain_db"), character_scale=prepare_report.get("character_scale_initial"))

    local_a2a.parent.mkdir(parents=True, exist_ok=True)
    _log(log_fp, "stage_start_write_a2a_float32", path=str(local_a2a))
    sf.write(str(local_a2a), np.asarray(staged, dtype=np.float32), sr, format="WAV", subtype="FLOAT")
    staged = None
    gc.collect()
    _log(log_fp, "stage_done_write_a2a_float32", path=str(local_a2a), bytes=local_a2a.stat().st_size)

    if args.supabase_a2a:
        _log(log_fp, "stage_start_a2a_upload", path=args.supabase_a2a)
        storage = SupabaseStorage.from_env()
        storage.upload_file(args.supabase_a2a, str(local_a2a), content_type="audio/wav")
        _log(log_fp, "stage_done_a2a_upload", path=args.supabase_a2a, bytes=local_a2a.stat().st_size)

    state.update({
        "state_schema": "busy_stage_a2a_v1",
        "stage": "A-2a",
        "stage_a2a_completed_at_utc": _now(),
        "mode": args.mode,
        "mastering_intensity": _effective_mastering_intensity(decision, getattr(args, "mastering_intensity", "auto")),
        "auto_intensity_policy": decision.get("auto_intensity_policy", {}),
        "sr": sr,
        # From here onward use the governed/attached decision and the post-pre-clean analysis.
        "decision": render_decision,
        "input_analysis": prepare_report.get("input_analysis", features),
        "original_input_analysis_before_pre_clean": prepare_report.get("original_input_analysis_before_pre_clean", {}),
        "residue_pre_clean_stage": prepare_report.get("residue_pre_clean_stage", {}),
        "pre_limiter_prepare_report": prepare_report,
        "a2a_format": "float32_wav",
        "a2a_supabase_path": args.supabase_a2a,
    })
    if args.stage_state:
        _write_json_file(args.stage_state, state)
        _log(log_fp, "stage_done_write_stage_state", path=args.stage_state, has_analysis_sources=bool(state.get("analysis_sources")), has_realtime_audio_judge=bool(((state.get("input_analysis") or {}).get("realtime_audio_judge") if isinstance(state.get("input_analysis"), dict) else None)), has_essentia_tensorflow=bool((((state.get("input_analysis") or {}).get("essentia_evidence") or {}).get("tensorflow_tags") if isinstance((state.get("input_analysis") or {}).get("essentia_evidence"), dict) else None)) if isinstance(state.get("input_analysis"), dict) else False)
        _upload_stage_state_if_requested(args, log_fp)
    Path(args.report).write_text(json.dumps(prepare_report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _log(log_fp, "worker_stage_pre_prepare_done", elapsed_sec=round(time.time() - t0, 3))
    return 0


def run_stage_pre_render(args: argparse.Namespace, log_fp) -> int:
    """Stage A-2b: render pre-limiter master from A-2a checkpoint."""
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    local_a2a = Path(args.a2a or (work_dir / "a2a_prepared_float32.wav"))
    local_intermediate = Path(args.intermediate or (work_dir / "pre_limiter_float32.wav"))
    _download_stage_state_if_requested(args, log_fp)
    if not args.stage_state or not Path(args.stage_state).exists():
        raise RuntimeError("--stage-state is required and must exist for Stage A-2b")
    state = json.loads(Path(args.stage_state).read_text(encoding="utf-8"))
    decision = _canonicalize_decision_for_stage(state.get("decision", {}), stage="A-2b_state_loaded")
    state["decision"] = decision
    state["decision_digest"] = _decision_profile_digest(decision)
    features = state.get("input_analysis", {})
    prepare_report = state.get("pre_limiter_prepare_report", {})
    _log(log_fp, "stage_state_decision_loaded", **_decision_profile_digest(decision), state_job_id=state.get("job_id"), path=args.stage_state)
    _apply_intensity_from_stage_state(state, args, log_fp, phase="A-2b_state_load")
    if not decision.get("selected_profile"):
        raise RuntimeError("Stage state is missing selected_profile; refusing to continue with unknown mastering profile")
    if args.supabase_a2a:
        _log(log_fp, "stage_start_a2a_download", path=args.supabase_a2a)
        storage = SupabaseStorage.from_env()
        storage.download_file(args.supabase_a2a, str(local_a2a))
        _log(log_fp, "stage_done_a2a_download", local=str(local_a2a), bytes=local_a2a.stat().st_size)
    _log(log_fp, "stage_start_load_a2a_float32")
    staged, sr = load_audio_file(str(local_a2a), target_sr=48_000, dtype="float32")
    _log(log_fp, "stage_done_load_a2a_float32", sr=sr, shape=list(staged.shape), dtype=str(staged.dtype), seconds=round(staged.shape[0] / sr, 3))

    _log(log_fp, "stage_start_pre_limiter_render")
    pre_limiter, pre_report = render_pre_limiter_from_prepared_stage(staged, sr, decision, prepare_report, mode=args.mode)
    _attach_stage_m_to_report(pre_report, features=features, decision=decision, state=state)
    staged = None
    gc.collect()
    _log(log_fp, "stage_done_pre_limiter_render", output_shape=list(pre_limiter.shape), pre_limiter_lufs=pre_report.get("pre_limiter_analysis_fast", {}).get("integrated_lufs"), pre_limiter_true_peak=pre_report.get("pre_limiter_analysis_fast", {}).get("approx_true_peak_dbfs"))

    local_intermediate.parent.mkdir(parents=True, exist_ok=True)
    _log(log_fp, "stage_start_write_intermediate_float32", path=str(local_intermediate))
    sf.write(str(local_intermediate), np.asarray(pre_limiter, dtype=np.float32), sr, format="WAV", subtype="FLOAT")
    pre_limiter = None
    gc.collect()
    _log(log_fp, "stage_done_write_intermediate_float32", path=str(local_intermediate), bytes=local_intermediate.stat().st_size)
    if args.supabase_intermediate:
        _log(log_fp, "stage_start_intermediate_upload", path=args.supabase_intermediate)
        storage = SupabaseStorage.from_env()
        storage.upload_file(args.supabase_intermediate, str(local_intermediate), content_type="audio/wav")
        _log(log_fp, "stage_done_intermediate_upload", path=args.supabase_intermediate, bytes=local_intermediate.stat().st_size)

    state.update({
        "state_schema": "busy_stage_a2b_v1",
        "stage": "A-2b",
        "stage_a2b_completed_at_utc": _now(),
        "mode": args.mode,
        "mastering_intensity": _effective_mastering_intensity(decision, getattr(args, "mastering_intensity", "auto")),
        "auto_intensity_policy": decision.get("auto_intensity_policy", {}),
        "sr": sr,
        "decision": decision,
        "input_analysis": pre_report.get("input_analysis", features),
        "original_input_analysis_before_pre_clean": pre_report.get("original_input_analysis_before_pre_clean", state.get("original_input_analysis_before_pre_clean", {})),
        "residue_pre_clean_stage": pre_report.get("residue_pre_clean_stage", state.get("residue_pre_clean_stage", {})),
        "pre_clean_plan": (features if isinstance(features, dict) else {}).get("pre_clean_plan", state.get("pre_clean_plan", {})),
        "post_pre_clean_analysis": (features if isinstance(features, dict) else {}).get("post_pre_clean_analysis", state.get("post_pre_clean_analysis", {})),
        "mix_conditioning_plan": (features if isinstance(features, dict) else {}).get("mix_conditioning_plan", state.get("mix_conditioning_plan", {})),
        "loudness_feasibility_predictor": (features if isinstance(features, dict) else {}).get("loudness_feasibility_predictor", state.get("loudness_feasibility_predictor", {})),
        "perceptual_quality_score_baseline": (features if isinstance(features, dict) else {}).get("perceptual_quality_score_baseline", state.get("perceptual_quality_score_baseline", {})),
        "multi_criteria_stress_window_requirements": (features if isinstance(features, dict) else {}).get("multi_criteria_stress_window_requirements", state.get("multi_criteria_stress_window_requirements", {})),
        "multi_component_assist": (features if isinstance(features, dict) else {}).get("multi_component_assist", state.get("multi_component_assist", {})),
        "multi_component_assist_summary": (features if isinstance(features, dict) else {}).get("multi_component_assist_summary", state.get("multi_component_assist_summary", {})),
        "stem_evidence_assist": (features if isinstance(features, dict) else {}).get("stem_evidence_assist", state.get("stem_evidence_assist", {})),
        "stage_m_stem_premaster_builder": (features if isinstance(features, dict) else {}).get("stage_m_stem_premaster_builder", state.get("stage_m_stem_premaster_builder", {})),
        "busy_auto_mixing": (features if isinstance(features, dict) else {}).get("busy_auto_mixing", state.get("busy_auto_mixing", {})),
        "busy_auto_mixing_summary": (features if isinstance(features, dict) else {}).get("busy_auto_mixing_summary", state.get("busy_auto_mixing_summary", {})),
        "v6370_body_center_vocal_support": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6370_body_center_vocal_support", state.get("v6370_body_center_vocal_support", {}))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("v6370_body_center_vocal_support", {}),
        "v6380_bass_harmonic_translation": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", {})))))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", {})),
        "bass_harmonic_translation": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", {})))))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", {})),
        "v6390_side_texture_stereo_cleanliness": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6390_side_texture_stereo_cleanliness", (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("side_texture_stereo_cleanliness", state.get("v6390_side_texture_stereo_cleanliness", state.get("side_texture_stereo_cleanliness", {})))))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("v6390_side_texture_stereo_cleanliness", state.get("side_texture_stereo_cleanliness", {})),
        "side_texture_control": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("side_texture_control", state.get("side_texture_control", {}))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("side_texture_control", {}),
        "low_grs_clean_component_assist": (features if isinstance(features, dict) else {}).get("low_grs_clean_component_assist", state.get("low_grs_clean_component_assist", {})),
        "blocker_remediation_plan": (features if isinstance(features, dict) else {}).get("blocker_remediation_plan", state.get("blocker_remediation_plan", {})),
        "post_remediation_predicted_metrics": (features if isinstance(features, dict) else {}).get("post_remediation_predicted_metrics", state.get("post_remediation_predicted_metrics", {})),
        "optimized_pipeline_architecture": (features if isinstance(features, dict) else {}).get("optimized_pipeline_architecture", state.get("optimized_pipeline_architecture", {})),
        "render_budget_policy": decision.get("render_budget_policy", state.get("render_budget_policy", {})),
        "analysis_sources": _analysis_sources_snapshot(pre_report.get("input_analysis", features), decision),
        "pre_limiter_report": pre_report,
        "intermediate_format": "float32_wav",
        "intermediate_supabase_path": args.supabase_intermediate,
    })
    if args.stage_state:
        _write_json_file(args.stage_state, state)
        _log(log_fp, "stage_done_write_stage_state", path=args.stage_state, has_analysis_sources=bool(state.get("analysis_sources")), has_realtime_audio_judge=bool(((state.get("input_analysis") or {}).get("realtime_audio_judge") if isinstance(state.get("input_analysis"), dict) else None)), has_essentia_tensorflow=bool((((state.get("input_analysis") or {}).get("essentia_evidence") or {}).get("tensorflow_tags") if isinstance((state.get("input_analysis") or {}).get("essentia_evidence"), dict) else None)) if isinstance(state.get("input_analysis"), dict) else False)
        _upload_stage_state_if_requested(args, log_fp)
    Path(args.report).write_text(json.dumps(pre_report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _log(log_fp, "worker_stage_pre_render_done", elapsed_sec=round(time.time() - t0, 3))
    return 0

def run_stage_pre(args: argparse.Namespace, log_fp) -> int:
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    local_input = Path(args.input) if args.input else work_dir / "input_from_supabase.wav"
    local_preprocessed = Path(args.preprocessed or (work_dir / "preprocessed_float32.wav"))
    local_intermediate = Path(args.intermediate or (work_dir / "pre_limiter_float32.wav"))

    # v8.4+ Stage A-2 path: reload both A-1 state and preprocessed audio
    # from Supabase in a fresh process, then exit before the limiter.
    if args.supabase_preprocessed:
        _download_stage_state_if_requested(args, log_fp)
    if args.stage_state and Path(args.stage_state).exists() and (args.supabase_preprocessed or args.preprocessed):
        state_in = json.loads(Path(args.stage_state).read_text(encoding="utf-8"))
        decision = _canonicalize_decision_for_stage(state_in.get("decision", {}), stage="A-2_legacy_state_loaded")
        state_in["decision"] = decision
        state_in["decision_digest"] = _decision_profile_digest(decision)
        features = state_in.get("input_analysis", {})
        _log(log_fp, "stage_state_decision_loaded", **_decision_profile_digest(decision), state_job_id=state_in.get("job_id"), path=args.stage_state)
        _apply_intensity_from_stage_state(state_in, args, log_fp, phase="A-2_legacy_state_load")
        if args.supabase_preprocessed:
            _log(log_fp, "stage_start_preprocessed_download", path=args.supabase_preprocessed)
            storage = SupabaseStorage.from_env()
            storage.download_file(args.supabase_preprocessed, str(local_preprocessed))
            _log(log_fp, "stage_done_preprocessed_download", local=str(local_preprocessed), bytes=local_preprocessed.stat().st_size)
        _log(log_fp, "stage_start_load_preprocessed_float32")
        y, sr = load_audio_file(str(local_preprocessed), target_sr=48_000, dtype="float32")
        _log(log_fp, "stage_done_load_preprocessed_float32", sr=sr, shape=list(y.shape), dtype=str(y.dtype), seconds=round(y.shape[0] / sr, 3))
    else:
        # Backward-compatible v7.5 path.
        # v8.5.3.45 hotfix: the normal Cloud Run orchestrator still enters this
        # legacy ``pre`` stage directly.  v41-v43/44 added multi-component ZIP
        # handling to run_single_stage/run_stage_analyze, but this path was still
        # building the mastering decision from the original WAV only.  Download and
        # analyze the optional separated-track ZIP here as well, then optionally
        # rebuild the effective input with the residual-preserving premaster before
        # the pre-limiter render.
        if args.supabase_input:
            _log(log_fp, "stage_start_supabase_download", path=args.supabase_input)
            storage = SupabaseStorage.from_env()
            storage.download_file(args.supabase_input, str(local_input))
            _log(log_fp, "stage_done_supabase_download", local=str(local_input), bytes=local_input.stat().st_size)

        local_stem_zip = _download_optional_multicomponent_zip(args, work_dir, log_fp)
        premaster_output_path = None  # v62.2: legacy residual stem-assisted premaster is not generated
        # v62.4.1: keep Stage M after original analysis even in the legacy pre-stage path.
        y, sr, features, decision, _preflight = _build_features_and_decision(local_input, args.private_zip, args.mode, log_fp, args.user_style_hint, args.private_rulepack, stem_zip_path=local_stem_zip, work_dir=work_dir)
        # v8.5.3.62.9.7: Stage M mixdown is removed in the legacy pre-stage path too.
        assist = features.get("multi_component_assist") if isinstance(features.get("multi_component_assist"), dict) else {
            "provided": bool(local_stem_zip),
            "enabled": _env_on("BUSY_STEM_EVIDENCE_ASSIST", "1"),
            "usable": False,
            "recommended_mode": "stem_evidence_assist_unavailable",
            "separated_tracks_used_for_audio_render": False,
        }
        stage_m_removed = _build_stage_m_stem_premaster_report(assist, stem_zip_provided=bool(local_stem_zip))
        _attach_stage_m_report(features, decision, assist, stage_m_removed)
        _log(log_fp, "stage_m_removed_original_stereo_route", stem_zip_provided=bool(local_stem_zip), reason=stage_m_removed.get("reason"), mode=stage_m_removed.get("mode"), selected_source=stage_m_removed.get("selected_source"))
        y, sr, features, decision, _preflight = _apply_busy_auto_mixing_if_requested(y=y, sr=sr, features=features, decision=decision, preflight_report=_preflight, local_input=local_input, local_stem_zip=local_stem_zip, args=args, log_fp=log_fp, work_dir=work_dir)

    _log(log_fp, "stage_start_pre_limiter_master")
    pre_limiter, pre_report = render_pre_limiter_master(y, sr, decision, mode=args.mode, pre_analysis=features)
    _attach_stage_m_to_report(pre_report, features=features, decision=decision, state=locals().get("state_in") if "state_in" in locals() else locals().get("state"))
    _attach_busy_auto_mixing_to_report(pre_report, features=features, decision=decision, state=locals().get("state_in") if "state_in" in locals() else locals().get("state"))
    y = None
    gc.collect()
    _log(log_fp, "stage_done_pre_limiter_master", output_shape=list(pre_limiter.shape), pre_limiter_lufs=pre_report.get("pre_limiter_analysis_fast", {}).get("integrated_lufs"), pre_limiter_true_peak=pre_report.get("pre_limiter_analysis_fast", {}).get("approx_true_peak_dbfs"))
    local_intermediate.parent.mkdir(parents=True, exist_ok=True)
    _log(log_fp, "stage_start_write_intermediate_float32", path=str(local_intermediate))
    sf.write(str(local_intermediate), np.asarray(pre_limiter, dtype=np.float32), sr, format="WAV", subtype="FLOAT")
    pre_limiter = None
    gc.collect()
    _log(log_fp, "stage_done_write_intermediate_float32", path=str(local_intermediate), bytes=local_intermediate.stat().st_size)
    if args.supabase_intermediate:
        _log(log_fp, "stage_start_intermediate_upload", path=args.supabase_intermediate)
        storage = SupabaseStorage.from_env()
        storage.upload_file(args.supabase_intermediate, str(local_intermediate), content_type="audio/wav")
        _log(log_fp, "stage_done_intermediate_upload", path=args.supabase_intermediate, bytes=local_intermediate.stat().st_size)
    state = {}
    if args.stage_state and Path(args.stage_state).exists():
        try:
            state = json.loads(Path(args.stage_state).read_text(encoding="utf-8"))
        except Exception:
            state = {}
    state.update({
        "state_schema": "busy_stage_a2_v1",
        "stage": "A-2",
        "stage_a2_completed_at_utc": _now(),
        "mode": args.mode,
        "mastering_intensity": _effective_mastering_intensity(decision, getattr(args, "mastering_intensity", "auto")),
        "auto_intensity_policy": decision.get("auto_intensity_policy", {}),
        "sr": sr,
        "decision": decision,
        "input_analysis": pre_report.get("input_analysis", features),
        "original_input_analysis_before_pre_clean": pre_report.get("original_input_analysis_before_pre_clean", state.get("original_input_analysis_before_pre_clean", {})),
        "residue_pre_clean_stage": pre_report.get("residue_pre_clean_stage", state.get("residue_pre_clean_stage", {})),
        "pre_clean_plan": (features if isinstance(features, dict) else {}).get("pre_clean_plan", state.get("pre_clean_plan", {})),
        "post_pre_clean_analysis": (features if isinstance(features, dict) else {}).get("post_pre_clean_analysis", state.get("post_pre_clean_analysis", {})),
        "mix_conditioning_plan": (features if isinstance(features, dict) else {}).get("mix_conditioning_plan", state.get("mix_conditioning_plan", {})),
        "loudness_feasibility_predictor": (features if isinstance(features, dict) else {}).get("loudness_feasibility_predictor", state.get("loudness_feasibility_predictor", {})),
        "perceptual_quality_score_baseline": (features if isinstance(features, dict) else {}).get("perceptual_quality_score_baseline", state.get("perceptual_quality_score_baseline", {})),
        "multi_criteria_stress_window_requirements": (features if isinstance(features, dict) else {}).get("multi_criteria_stress_window_requirements", state.get("multi_criteria_stress_window_requirements", {})),
        "multi_component_assist": (features if isinstance(features, dict) else {}).get("multi_component_assist", state.get("multi_component_assist", {})),
        "multi_component_assist_summary": (features if isinstance(features, dict) else {}).get("multi_component_assist_summary", state.get("multi_component_assist_summary", {})),
        "stem_evidence_assist": (features if isinstance(features, dict) else {}).get("stem_evidence_assist", state.get("stem_evidence_assist", {})),
        "stage_m_stem_premaster_builder": (features if isinstance(features, dict) else {}).get("stage_m_stem_premaster_builder", state.get("stage_m_stem_premaster_builder", {})),
        "busy_auto_mixing": (features if isinstance(features, dict) else {}).get("busy_auto_mixing", state.get("busy_auto_mixing", {})),
        "busy_auto_mixing_summary": (features if isinstance(features, dict) else {}).get("busy_auto_mixing_summary", state.get("busy_auto_mixing_summary", {})),
        "v6370_body_center_vocal_support": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6370_body_center_vocal_support", state.get("v6370_body_center_vocal_support", {}))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("v6370_body_center_vocal_support", {}),
        "v6380_bass_harmonic_translation": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", {})))))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", {})),
        "bass_harmonic_translation": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", {})))))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", {})),
        "v6390_side_texture_stereo_cleanliness": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6390_side_texture_stereo_cleanliness", (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("side_texture_stereo_cleanliness", state.get("v6390_side_texture_stereo_cleanliness", state.get("side_texture_stereo_cleanliness", {})))))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("v6390_side_texture_stereo_cleanliness", state.get("side_texture_stereo_cleanliness", {})),
        "side_texture_control": (((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("side_texture_control", state.get("side_texture_control", {}))) if isinstance((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}), dict) else state.get("side_texture_control", {}),
        "busy_auto_mixing_selected_recipe": ((features if isinstance(features, dict) else {}).get("busy_auto_mixing", {}) or {}).get("selected_mix_recipe", state.get("busy_auto_mixing_selected_recipe")),
        "low_grs_clean_component_assist": (features if isinstance(features, dict) else {}).get("low_grs_clean_component_assist", state.get("low_grs_clean_component_assist", {})),
        "blocker_remediation_plan": (features if isinstance(features, dict) else {}).get("blocker_remediation_plan", state.get("blocker_remediation_plan", {})),
        "post_remediation_predicted_metrics": (features if isinstance(features, dict) else {}).get("post_remediation_predicted_metrics", state.get("post_remediation_predicted_metrics", {})),
        "optimized_pipeline_architecture": (features if isinstance(features, dict) else {}).get("optimized_pipeline_architecture", state.get("optimized_pipeline_architecture", {})),
        "render_budget_policy": decision.get("render_budget_policy", state.get("render_budget_policy", {})),
        "analysis_sources": _analysis_sources_snapshot(pre_report.get("input_analysis", features), decision),
        "pre_limiter_report": pre_report,
        "intermediate_format": "float32_wav",
        "intermediate_supabase_path": args.supabase_intermediate,
    })
    if args.stage_state:
        _write_json_file(args.stage_state, state)
        _log(log_fp, "stage_done_write_stage_state", path=args.stage_state, has_analysis_sources=bool(state.get("analysis_sources")), has_realtime_audio_judge=bool(((state.get("input_analysis") or {}).get("realtime_audio_judge") if isinstance(state.get("input_analysis"), dict) else None)), has_essentia_tensorflow=bool((((state.get("input_analysis") or {}).get("essentia_evidence") or {}).get("tensorflow_tags") if isinstance((state.get("input_analysis") or {}).get("essentia_evidence"), dict) else None)) if isinstance(state.get("input_analysis"), dict) else False)
        _upload_stage_state_if_requested(args, log_fp)
    Path(args.report).write_text(json.dumps(pre_report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    _log(log_fp, "worker_stage_pre_done", elapsed_sec=round(time.time() - t0, 3))
    return 0


def run_stage_limit(args: argparse.Namespace, log_fp) -> int:
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    local_intermediate = Path(args.intermediate or (work_dir / "pre_limiter_float32.wav"))
    local_output = Path(args.output) if args.output else work_dir / "output_to_supabase.wav"
    if args.supabase_intermediate:
        _log(log_fp, "stage_start_intermediate_download", path=args.supabase_intermediate)
        storage = SupabaseStorage.from_env()
        storage.download_file(args.supabase_intermediate, str(local_intermediate))
        _log(log_fp, "stage_done_intermediate_download", local=str(local_intermediate), bytes=local_intermediate.stat().st_size)
    if not args.stage_state:
        raise RuntimeError("--stage-state is required for limiter stage")
    _download_stage_state_if_requested(args, log_fp)
    state = json.loads(Path(args.stage_state).read_text(encoding="utf-8"))
    decision = _canonicalize_decision_for_stage(state.get("decision", {}), stage="B_state_loaded")
    before = state.get("input_analysis", {})
    # v62.9.9.2: ensure the research-governance bundle survives the two-stage
    # state handoff.  v62.9.9.1 attached it inside the render path, which could
    # leave Stage B reports with empty top-level governance dictionaries.
    decision = _attach_v6299_research_governance_if_missing(before if isinstance(before, dict) else {}, decision, mode=args.mode, stage="B_state_loaded_before_finalize", log_fp=log_fp)
    before, decision, _v6360_fw = _attach_v6360_research_capability_framework_if_missing(
        before if isinstance(before, dict) else {},
        decision,
        stage="B_state_loaded_before_finalize",
        report=state,
        log_fp=log_fp,
    )
    state["input_analysis"] = before
    state["decision"] = decision
    state["decision_digest"] = _decision_profile_digest(decision)
    pre_limiter_report = state.get("pre_limiter_report", {})
    _log(log_fp, "stage_state_decision_loaded", **_decision_profile_digest(decision), state_job_id=state.get("job_id"), path=args.stage_state)
    _apply_intensity_from_stage_state(state, args, log_fp, phase="B_state_load")
    if not decision.get("selected_profile"):
        raise RuntimeError("Stage state is missing selected_profile; refusing to continue with unknown mastering profile")
    _log(log_fp, "stage_start_load_intermediate_float32")
    pre_limiter, sr = load_audio_file(str(local_intermediate), target_sr=48_000, dtype="float32")
    _log(log_fp, "stage_done_load_intermediate_float32", sr=sr, shape=list(pre_limiter.shape), dtype=str(pre_limiter.dtype), seconds=round(pre_limiter.shape[0] / sr, 3))
    state.update({
        "state_schema": "busy_stage_b_running_v1",
        "stage": "B-running",
        "stage_b_started_at_utc": _now(),
        "mode": args.mode,
        "mastering_intensity": _effective_mastering_intensity(decision, getattr(args, "mastering_intensity", "auto")),
        "auto_intensity_policy": decision.get("auto_intensity_policy", {}),
        "sr": sr,
        "decision": decision,
        "decision_digest": _decision_profile_digest(decision),
        "analysis_sources": _analysis_sources_snapshot(before if isinstance(before, dict) else {}, decision),
        "optimized_pipeline_architecture": state.get("optimized_pipeline_architecture", decision.get("optimized_processing_order", {})),
        "render_budget_policy": state.get("render_budget_policy", decision.get("render_budget_policy", {})),
        "research_capability_framework": (before if isinstance(before, dict) else {}).get("research_capability_framework", decision.get("research_capability_framework", state.get("research_capability_framework", {}))),
        "research_capability_summary": compact_research_capability_summary((before if isinstance(before, dict) else {}).get("research_capability_framework", decision.get("research_capability_framework", state.get("research_capability_framework", {})))),
        "remaining_capability_matrix": (before if isinstance(before, dict) else {}).get("remaining_capability_matrix", decision.get("remaining_capability_matrix", state.get("remaining_capability_matrix", []))),
        "capability_router": (before if isinstance(before, dict) else {}).get("capability_router", decision.get("capability_router", state.get("capability_router", {}))),
        "capability_contracts": (before if isinstance(before, dict) else {}).get("capability_contracts", decision.get("capability_contracts", state.get("capability_contracts", {}))),
        "capability_conflict_policy": (before if isinstance(before, dict) else {}).get("capability_conflict_policy", decision.get("capability_conflict_policy", state.get("capability_conflict_policy", {}))),
        "multi_component_assist": state.get("multi_component_assist", (before if isinstance(before, dict) else {}).get("multi_component_assist", {})),
        "multi_component_assist_summary": state.get("multi_component_assist_summary", (before if isinstance(before, dict) else {}).get("multi_component_assist_summary", {})),
        "stem_evidence_assist": state.get("stem_evidence_assist", (before if isinstance(before, dict) else {}).get("stem_evidence_assist", {})),
        "busy_auto_mixing": state.get("busy_auto_mixing", (before if isinstance(before, dict) else {}).get("busy_auto_mixing", {})),
        "busy_auto_mixing_summary": state.get("busy_auto_mixing_summary", (before if isinstance(before, dict) else {}).get("busy_auto_mixing_summary", {})),
        "busy_auto_mixing_selected_recipe": (state.get("busy_auto_mixing_selected_recipe") or ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("selected_mix_recipe")),
        "stage_b_runtime_guard": {
            "stage_b_timeout_sec": int(os.environ.get("BUSY_STAGE_B_TIMEOUT_SEC", "900") or "900"),
            "blocker_resolution_wall_time_sec": float(os.environ.get("BUSY_BLOCKER_RESOLUTION_WALL_TIME_SEC", "360") or "360"),
            "blocker_resolution_max_steps": int(os.environ.get("BUSY_BLOCKER_RESOLUTION_SAFETY_MAX_STEPS", "18") or "18"),
            "policy": "intermediate heartbeat before limiter/planner so stuck jobs are distinguishable from A-2 completion",
        },
    })
    if args.stage_state:
        _write_json_file(args.stage_state, state)
        _log(log_fp, "stage_done_write_stage_state", path=args.stage_state, stage="B-running")
        _upload_stage_state_if_requested(args, log_fp)
    _log(log_fp, "stage_start_fresh_limiter_master")
    final, report = finalize_limiter_master(pre_limiter, sr, decision, before=before, pre_limiter_report=pre_limiter_report, mode=args.mode)
    _attach_stage_m_to_report(report, features=before, decision=decision, state=state)
    _attach_busy_auto_mixing_to_report(report, features=before, decision=decision, state=state)
    report = _v6314_4_finalize_floor_tolerance_and_handoff_markers(report, decision)
    report = _v6362_8_finalize_report_authority(report, decision)
    pre_limiter = None
    gc.collect()
    _log(log_fp, "stage_done_fresh_limiter_master", output_shape=list(final.shape), output_lufs=report.get("output_analysis", {}).get("integrated_lufs"), output_true_peak=report.get("output_analysis", {}).get("approx_true_peak_dbfs"), oversample_factor=report.get("oversample_factor"))
    _log(log_fp, "stage_start_write_wav")
    write_wav_pcm24(str(local_output), final, sr=48_000)
    final = None
    gc.collect()
    _log(log_fp, "stage_done_write_wav", output=str(local_output))
    if args.supabase_output:
        _log(log_fp, "stage_start_supabase_upload", path=args.supabase_output)
        storage = SupabaseStorage.from_env()
        storage.upload_file(args.supabase_output, str(local_output), content_type="audio/wav")
        _log(log_fp, "stage_done_supabase_upload", path=args.supabase_output, bytes=local_output.stat().st_size)
    report = _force_report_to_stage_decision(report, state, decision, stage="B_report_finalized")
    _attach_stage_m_to_report(report, features=before, decision=decision, state=state)
    _attach_busy_auto_mixing_to_report(report, features=before, decision=decision, state=state)
    report = _v6314_4_finalize_floor_tolerance_and_handoff_markers(report, decision)
    report = _v6362_8_finalize_report_authority(report, decision)
    report = attach_research_capability_to_report(report, features=before, decision=decision, stage="B_report_finalized")
    # v63.6.2.11: framework attachment can run after the DML authority pass;
    # DML report authority must be the last schema/source stamp before v64.0
    # finishing intelligence attaches a report-only overlay.
    report = _v6362_8_finalize_report_authority(report, decision)
    report = attach_finishing_intelligence_to_report(report, state=state, decision=decision, output_path=local_output)
    report = attach_research_capability_to_report(report, features=before, decision=decision, stage="B_v6400_final_report")
    state.update({
        "state_schema": "busy_stage_b_v1",
        "stage": "B",
        "stage_b_completed_at_utc": _now(),
        "mode": args.mode,
        "mastering_intensity": _effective_mastering_intensity(decision, getattr(args, "mastering_intensity", "auto")),
        "auto_intensity_policy": decision.get("auto_intensity_policy", {}),
        "sr": sr,
        "decision": decision,
        "decision_digest": _decision_profile_digest(decision),
        "analysis_sources": _analysis_sources_snapshot(before if isinstance(before, dict) else {}, decision),
        "optimized_pipeline_architecture": state.get("optimized_pipeline_architecture", report.get("optimized_pipeline_architecture", decision.get("optimized_processing_order", {}))),
        "render_budget_policy": state.get("render_budget_policy", report.get("render_budget_policy", decision.get("render_budget_policy", {}))),
        "research_capability_framework": report.get("research_capability_framework", (before if isinstance(before, dict) else {}).get("research_capability_framework", decision.get("research_capability_framework", state.get("research_capability_framework", {})))),
        "research_capability_summary": compact_research_capability_summary(report.get("research_capability_framework", (before if isinstance(before, dict) else {}).get("research_capability_framework", decision.get("research_capability_framework", state.get("research_capability_framework", {}))))),
        "remaining_capability_matrix": report.get("remaining_capability_matrix", (before if isinstance(before, dict) else {}).get("remaining_capability_matrix", decision.get("remaining_capability_matrix", state.get("remaining_capability_matrix", [])))),
        "capability_router": report.get("capability_router", (before if isinstance(before, dict) else {}).get("capability_router", decision.get("capability_router", state.get("capability_router", {})))),
        "capability_contracts": report.get("capability_contracts", (before if isinstance(before, dict) else {}).get("capability_contracts", decision.get("capability_contracts", state.get("capability_contracts", {})))),
        "capability_conflict_policy": report.get("capability_conflict_policy", (before if isinstance(before, dict) else {}).get("capability_conflict_policy", decision.get("capability_conflict_policy", state.get("capability_conflict_policy", {})))),
        "v6361_drum_transient_punch": report.get("v6361_drum_transient_punch", report.get("drum_transient_punch", {})),
        "drum_transient_punch": report.get("drum_transient_punch", report.get("v6361_drum_transient_punch", {})),
        **_v6360_2_stage_state_lineage_fields(report, state, stage="B_report_finalized"),
        **_v6362_11_stage_state_authority_mirror_fields(report),
        "multi_component_assist": state.get("multi_component_assist", (before if isinstance(before, dict) else {}).get("multi_component_assist", {})),
        "multi_component_assist_summary": state.get("multi_component_assist_summary", (before if isinstance(before, dict) else {}).get("multi_component_assist_summary", {})),
        "stem_evidence_assist": state.get("stem_evidence_assist", (before if isinstance(before, dict) else {}).get("stem_evidence_assist", {})),
        "stage_m_stem_premaster_builder": state.get("stage_m_stem_premaster_builder", (before if isinstance(before, dict) else {}).get("stage_m_stem_premaster_builder", report.get("stage_m_stem_premaster_builder", {}))),
        "busy_auto_mixing": state.get("busy_auto_mixing", (before if isinstance(before, dict) else {}).get("busy_auto_mixing", report.get("busy_auto_mixing", {}))),
        "busy_auto_mixing_summary": state.get("busy_auto_mixing_summary", (before if isinstance(before, dict) else {}).get("busy_auto_mixing_summary", report.get("busy_auto_mixing_summary", {}))),
        "v6370_body_center_vocal_support": report.get("v6370_body_center_vocal_support", state.get("v6370_body_center_vocal_support", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6370_body_center_vocal_support", {}))),
        "v6380_bass_harmonic_translation": report.get("v6380_bass_harmonic_translation", report.get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", {})))))),
        "bass_harmonic_translation": report.get("bass_harmonic_translation", report.get("v6380_bass_harmonic_translation", state.get("bass_harmonic_translation", state.get("v6380_bass_harmonic_translation", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("bass_harmonic_translation", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6380_bass_harmonic_translation", {})))))),
        "v6390_side_texture_stereo_cleanliness": report.get("v6390_side_texture_stereo_cleanliness", report.get("side_texture_stereo_cleanliness", state.get("v6390_side_texture_stereo_cleanliness", state.get("side_texture_stereo_cleanliness", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("v6390_side_texture_stereo_cleanliness", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("side_texture_stereo_cleanliness", {})))))),
        "side_texture_control": report.get("side_texture_control", state.get("side_texture_control", ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("side_texture_control", {}))),
        "v6400_finishing_intelligence": report.get("v6400_finishing_intelligence", state.get("v6400_finishing_intelligence", {})),
        "finishing_intelligence": report.get("finishing_intelligence", state.get("finishing_intelligence", {})),
        "v6404_commercial_maximum_density": report.get("v6404_commercial_maximum_density", state.get("v6404_commercial_maximum_density", {})),
        "commercial_maximum_density": report.get("commercial_maximum_density", state.get("commercial_maximum_density", {})),
        "commercial_density_candidate_lineage": (report.get("v6404_commercial_maximum_density", {}) or {}).get("candidates", []) if isinstance(report.get("v6404_commercial_maximum_density", {}), dict) else [],
        "clipper_limiter_candidate_lineage": [c for c in ((report.get("v6404_commercial_maximum_density", {}) or {}).get("candidates", []) if isinstance(report.get("v6404_commercial_maximum_density", {}), dict) else []) if isinstance(c, dict) and str(c.get("candidate_id")) == "clipper_limiter_candidate"],
        "maximum_push_policy": (report.get("v6404_commercial_maximum_density", {}) or {}).get("candidate_policy", "") if isinstance(report.get("v6404_commercial_maximum_density", {}), dict) else "",
        "warning_advisory_policy": (report.get("v6404_commercial_maximum_density", {}) or {}).get("warning_policy", {}) if isinstance(report.get("v6404_commercial_maximum_density", {}), dict) else {},
        "codec_safe_vs_maximum_wav_decision": (report.get("v6404_commercial_maximum_density", {}) or {}).get("codec_safe_vs_maximum_wav_decision", {}) if isinstance(report.get("v6404_commercial_maximum_density", {}), dict) else {},
        "playback_translation_auditor": report.get("playback_translation_auditor", state.get("playback_translation_auditor", {})),
        "codec_stress_test": report.get("codec_stress_test", state.get("codec_stress_test", {})),
        "shadow_master_candidate_selection": report.get("shadow_master_candidate_selection", state.get("shadow_master_candidate_selection", {})),
        "input_relative_damage_ownership": report.get("input_relative_damage_ownership", state.get("input_relative_damage_ownership", {})),
        "proxy_actual_calibration": report.get("proxy_actual_calibration", state.get("proxy_actual_calibration", {})),
        "decision_replay": report.get("decision_replay", state.get("decision_replay", {})),
        "corpus_regression": report.get("corpus_regression", state.get("corpus_regression", {})),
        "busy_auto_mixing_selected_recipe": (state.get("busy_auto_mixing_selected_recipe") or report.get("busy_auto_mixing_selected_recipe") or ((before if isinstance(before, dict) else {}).get("busy_auto_mixing", {}) or {}).get("selected_mix_recipe")),
        "final_report_summary": _stage_b_final_report_summary(report),
        **_stage_b_top_level_qc_fields(report),
    })
    if args.stage_state:
        _write_json_file(args.stage_state, state)
        _log(log_fp, "stage_done_write_stage_state", path=args.stage_state, has_analysis_sources=bool(state.get("analysis_sources")), has_post_master_qc=bool(state.get("post_master_qc_explainer")), has_ai_residue_architecture=bool(state.get("ai_residue_architecture_mapper")), stage="B")
        _upload_stage_state_if_requested(args, log_fp)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    debug_brief = _write_debug_brief_if_requested(args, report, state, log_fp)
    if isinstance(debug_brief, dict):
        report["debug_brief"] = debug_brief
        state["debug_brief"] = debug_brief
        _v6360_2_sync_stage_state_lineage(state, report, stage="B_debug_brief_bound")
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
        if args.stage_state:
            _write_json_file(args.stage_state, state)
            _upload_stage_state_if_requested(args, log_fp)
    _log(log_fp, "stage_done_report_bound_to_stage_state", **_decision_profile_digest(decision), has_stage_state_snapshot=bool(report.get("stage_state_snapshot")), has_analysis_sources=bool(report.get("analysis_sources")), has_debug_brief=bool((report.get("debug_brief") or {}).get("active") if isinstance(report.get("debug_brief"), dict) else False))
    _log(log_fp, "worker_stage_limit_done", elapsed_sec=round(time.time() - t0, 3), oversample_factor=report.get("oversample_factor"))
    return 0


def run_two_stage_orchestrator(args: argparse.Namespace, log_fp) -> int:
    t0 = time.time()
    work_dir = Path(args.report).resolve().parent
    state_path = work_dir / "two_stage_state.json"
    pre_report_path = work_dir / "pre_limiter_report.json"
    local_intermediate = work_dir / "pre_limiter_float32.wav"
    supabase_intermediate = args.supabase_intermediate or (_intermediate_path_from_output(args.supabase_output) if args.supabase_output else None)
    supabase_stage_state = args.supabase_stage_state
    # v8.5.3.32: the pre-limiter and limiter workers run inside the same Cloud Run
    # job directory.  Keeping the float32 intermediate local avoids one large
    # upload plus one large download while preserving the same split-worker memory
    # reset and final/stage-state uploads. Set BUSY_LOCAL_TWO_STAGE_INTERMEDIATE=0
    # to restore the old Supabase intermediate checkpoint behavior.
    local_two_stage_intermediate = _env_on("BUSY_LOCAL_TWO_STAGE_INTERMEDIATE", "1")
    effective_supabase_intermediate = None if local_two_stage_intermediate else supabase_intermediate
    _log(log_fp, "two_stage_orchestrator_start", supabase_intermediate=supabase_intermediate, effective_supabase_intermediate=effective_supabase_intermediate, local_two_stage_intermediate=local_two_stage_intermediate, supabase_stage_state=supabase_stage_state, requested_oversample=os.environ.get("BUSY_OVERSAMPLE", "16"), fallback_oversample=os.environ.get("BUSY_LIMITER_FALLBACK_OVERSAMPLE", "16"), safe_fallback=os.environ.get("BUSY_SAFE_FALLBACK_OVERSAMPLE", "8"))

    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--private-zip", args.private_zip,
        "--mode", args.mode,
        "--mastering-intensity", _normalize_mastering_intensity(getattr(args, "mastering_intensity", "auto")),
    ]
    if getattr(args, "supabase_stem_zip", ""):
        common += ["--supabase-stem-zip", args.supabase_stem_zip]
    if getattr(args, "stem_zip", ""):
        common += ["--stem-zip", args.stem_zip]
    if _busy_auto_mixing_requested(args):
        common += ["--busy-auto-mixing", "1"]
    if getattr(args, "private_rulepack", ""):
        common += ["--private-rulepack", args.private_rulepack]
    pre_cmd = common + [
        "--stage", "pre",
        "--report", str(pre_report_path),
        "--log", str(work_dir / "stage_pre.log"),
        "--stage-state", str(state_path),
        "--intermediate", str(local_intermediate),
    ]
    if args.supabase_input:
        pre_cmd += ["--supabase-input", args.supabase_input]
    elif args.input:
        pre_cmd += ["--input", args.input]
    if effective_supabase_intermediate:
        pre_cmd += ["--supabase-intermediate", effective_supabase_intermediate]
    if supabase_stage_state:
        pre_cmd += ["--supabase-stage-state", supabase_stage_state]

    rc = _run_child(pre_cmd, log_fp, "pre_limiter_worker", timeout_sec=int(os.environ.get("BUSY_STAGE_A_TIMEOUT_SEC", "1800")))
    if rc != 0:
        _log(log_fp, "two_stage_orchestrator_failed", failed_stage="pre", returncode=rc)
        return rc

    # The pre-limiter worker is gone now. Force garbage collection before final limiter worker starts.
    gc.collect()
    _log(log_fp, "two_stage_memory_reset_point", note="pre-limiter worker exited; starting fresh limiter worker")

    requested = str(os.environ.get("BUSY_OVERSAMPLE", "32"))
    fallback = str(os.environ.get("BUSY_LIMITER_FALLBACK_OVERSAMPLE", "16"))
    safe = str(os.environ.get("BUSY_SAFE_FALLBACK_OVERSAMPLE", "8"))
    attempts: list[str] = []
    for os_factor in [requested, fallback, safe]:
        if os_factor not in attempts:
            attempts.append(os_factor)

    last_rc = 1
    for idx, os_factor in enumerate(attempts, start=1):
        env = os.environ.copy()
        env["BUSY_OVERSAMPLE"] = str(os_factor)
        env["BUSY_TWO_STAGE_LIMITING"] = "1"
        # Let the fresh worker actually try the high OS factor unless the user disabled force.
        if str(os.environ.get("BUSY_TWO_STAGE_FORCE_LIMITER", "1")).lower() in {"1", "true", "yes", "on"}:
            env["BUSY_FORCE_OVERSAMPLE"] = "1"
        attempt_report = args.report if idx == 1 else str(Path(args.report).with_name(f"report_limiter_attempt_{os_factor}x.json"))
        limit_cmd = common + [
            "--stage", "limit",
            "--report", attempt_report,
            "--log", str(work_dir / f"stage_limit_{os_factor}x.log"),
            "--stage-state", str(state_path),
            "--intermediate", str(local_intermediate),
        ]
        if effective_supabase_intermediate:
            limit_cmd += ["--supabase-intermediate", effective_supabase_intermediate]
        if supabase_stage_state:
            limit_cmd += ["--supabase-stage-state", supabase_stage_state]
        if args.supabase_output:
            limit_cmd += ["--supabase-output", args.supabase_output]
        elif args.output:
            limit_cmd += ["--output", args.output]
        _log(log_fp, "two_stage_limiter_attempt_start", attempt=idx, oversample=os_factor)
        last_rc = _run_child(limit_cmd, log_fp, f"fresh_limiter_worker_{os_factor}x", timeout_sec=int(os.environ.get("BUSY_STAGE_B_TIMEOUT_SEC", "900")), env=env)
        if last_rc == 0:
            # Copy fallback attempt report to the canonical report path if needed.
            if attempt_report != args.report and Path(attempt_report).exists():
                Path(args.report).write_text(Path(attempt_report).read_text(encoding="utf-8"), encoding="utf-8")
            _log(log_fp, "two_stage_orchestrator_done", elapsed_sec=round(time.time() - t0, 3), limiter_oversample=os_factor)
            return 0
        _log(log_fp, "two_stage_limiter_attempt_failed", attempt=idx, oversample=os_factor, returncode=last_rc)
        gc.collect()
    _log(log_fp, "two_stage_orchestrator_failed", failed_stage="limit", returncode=last_rc)
    return last_rc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=False)
    parser.add_argument("--output", required=False)
    parser.add_argument("--supabase-input", required=False)
    parser.add_argument("--supabase-output", required=False)
    parser.add_argument("--supabase-intermediate", required=False)
    parser.add_argument("--supabase-a2a", required=False)
    parser.add_argument("--supabase-preprocessed", required=False)
    parser.add_argument("--supabase-stage-state", required=False)
    parser.add_argument("--supabase-stem-zip", required=False)
    parser.add_argument("--intermediate", required=False)
    parser.add_argument("--a2a", required=False)
    parser.add_argument("--preprocessed", required=False)
    parser.add_argument("--stage-state", required=False)
    parser.add_argument("--stem-zip", required=False)
    parser.add_argument("--busy-auto-mixing", default=os.environ.get("BUSY_AUTO_MIXING_REQUESTED", "0"))
    parser.add_argument("--stage", choices=["orchestrate", "analyze", "pre", "pre_prepare", "pre_render", "limit", "single"], default="orchestrate")
    parser.add_argument("--report", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--private-zip", default="private_docs_encrypted.zip")
    parser.add_argument("--private-rulepack", default="")
    parser.add_argument("--mode", default="Auto Commercial Master")
    parser.add_argument("--mastering-intensity", default=os.environ.get("BUSY_MASTERING_INTENSITY", "auto"))
    parser.add_argument("--user-style-hint", default="")
    args = parser.parse_args()
    requested_mastering_intensity = _normalize_mastering_intensity(args.mastering_intensity)
    _apply_mastering_intensity_env_to_process(requested_mastering_intensity)
    args.mastering_intensity = requested_mastering_intensity

    os.environ["BUSY_JOB_LOG"] = args.log
    log_fp = open(args.log, "w", encoding="utf-8")
    try:
        _log(log_fp, "worker_start", stage=args.stage, input=args.input, output=args.output, supabase_input=args.supabase_input, supabase_output=args.supabase_output, supabase_intermediate=args.supabase_intermediate, supabase_a2a=args.supabase_a2a, supabase_preprocessed=args.supabase_preprocessed, supabase_stage_state=args.supabase_stage_state, supabase_stem_zip=bool(getattr(args, "supabase_stem_zip", "")), local_stem_zip=bool(getattr(args, "stem_zip", "")), busy_auto_mixing=_busy_auto_mixing_requested(args), mastering_intensity=args.mastering_intensity, true_peak_ceiling=os.environ.get("BUSY_TRUE_PEAK_CEILING_DB"), commercial_reference_true_peak=os.environ.get("BUSY_COMMERCIAL_REFERENCE_TRUE_PEAK_CEILING_DB"), db_target_chase=os.environ.get("BUSY_COMMERCIAL_DB_TARGET_CHASE"), oversample=os.environ.get("BUSY_OVERSAMPLE", "16"), two_stage_limiting=_two_stage_enabled(), adaptive_quality=os.environ.get("BUSY_ADAPTIVE_QUALITY", "1"), user_style_hint_provided=bool(_clean_user_style_hint(args.user_style_hint or os.environ.get("BUSY_USER_STYLE_HINT"))), rollback_max_attempts=os.environ.get("BUSY_ROLLBACK_MAX_ATTEMPTS", "2"), qc_oversample=os.environ.get("BUSY_QC_OVERSAMPLE", "2"), python=sys.version, cwd=os.getcwd())
        if args.stage == "analyze":
            return run_stage_analyze(args, log_fp)
        if args.stage == "pre_prepare":
            return run_stage_pre_prepare(args, log_fp)
        if args.stage == "pre_render":
            return run_stage_pre_render(args, log_fp)
        if args.stage == "pre":
            return run_stage_pre(args, log_fp)
        if args.stage == "limit":
            return run_stage_limit(args, log_fp)
        if args.stage == "single" or not _two_stage_enabled() or (not args.supabase_input and not args.input):
            return run_single_stage(args, log_fp)
        return run_two_stage_orchestrator(args, log_fp)
    except Exception as exc:
        tb = traceback.format_exc()
        _log(log_fp, "worker_exception", error_type=type(exc).__name__, error=str(exc), traceback=tb)
        # v8.5.1.1: persist worker exceptions into stage_state when possible so Streamlit logs are not the only clue.
        try:
            if getattr(args, "stage_state", None):
                p = Path(args.stage_state)
                state = {}
                if p.exists():
                    try:
                        state = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        state = {}
                state.setdefault("errors", [])
                state["stage_error"] = {
                    "stage": args.stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback_tail": tb[-4000:],
                    "ts_utc": _now(),
                }
                state["errors"].append(state["stage_error"])
                _write_json_file(str(p), state)
                try:
                    _upload_stage_state_if_requested(args, log_fp)
                except Exception as upload_exc:
                    _log(log_fp, "worker_exception_stage_state_upload_failed", error=str(upload_exc)[:500])
        except Exception as state_exc:
            _log(log_fp, "worker_exception_stage_state_write_failed", error=str(state_exc)[:500])
        return 1
    finally:
        try:
            log_fp.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
