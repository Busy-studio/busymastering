from __future__ import annotations

"""Research Capability Framework for Busy Auto Mastering v64.0.

The framework remains the authority that routes recurring blockers to named
capabilities instead of one-off warning-label hotfixes.  v64.0.4 adds Commercial Maximum Density Push Foundation while preserving the v63.6.2 DML/ownership contract and the v63.6.1/v63.7/v63.8/v63.9 guarded DSP modules. Finishing intelligence evaluates playback translation, actual/proxy codec stress, shadow candidates, damage ownership, decision replay, and regression snapshots; v64.0.4 separately renders actual Commercial/Maximum WAV density candidates with warnings reported as advisory unless severe damage is measured.
"""

import copy
import math
import os
from typing import Any

SCHEMA_VERSION = "busy_research_capability_framework_v8_5_3_64_0_4"
ENGINE_VERSION = "busy_auto_mastering_v8_5_3_64_0_4_research_capability_framework_commercial_maximum_density_push_foundation"
REPORT_SCHEMA = "busy_master_report_v8_5_3_64_0_4_research_capability_framework_commercial_maximum_density_push_foundation"


_CAPABILITY_REGISTRY: tuple[dict[str, Any], ...] = (
    {
        "id": "research_capability_framework",
        "title": "Research Capability Framework / Remaining Capability Matrix",
        "target_patch": "v63.6.0",
        "planned_status": "implemented",
        "source_basis": ["quality_upgrade_roadmap", "engine_expansion_report", "bamix_premaster_rules"],
        "telemetry_keys": ["research_capability_framework", "remaining_capability_matrix", "capability_router", "capability_contracts"],
        "complete_when": [
            "all_remaining_research_capabilities_are_explicitly_classified",
            "router_guard_rollback_telemetry_contracts_are_visible_in_stage_state_and_report",
            "ownership_dml_bamix_final_render_conflicts_are_declared",
        ],
    },
    {
        "id": "mix_conditioning_foundation",
        "title": "Stemless Mix Conditioning Foundation",
        "target_patch": "v8.5.3.40+",
        "planned_status": "implemented",
        "source_basis": ["stereo_mix_conditioning_report", "quality_upgrade_roadmap"],
        "telemetry_keys": ["mix_conditioning_plan", "loudness_feasibility_predictor", "perceptual_quality_score_baseline", "multi_criteria_stress_window_requirements"],
        "complete_when": [
            "parameterized_planning_after_pre_clean_before_blocker_scan",
            "no_intermediate_wav_export",
            "bounded_risk_caps_for_headroom_low_side_sibilance_side_high_transient_fragility",
        ],
    },
    {
        "id": "input_relative_damage_governance",
        "title": "Input-Relative Damage / Adaptive QC Governance",
        "target_patch": "v62.9.9.3+",
        "planned_status": "implemented",
        "source_basis": ["input_relative_damage_criteria_report"],
        "telemetry_keys": ["adaptive_input_relative_qc_governance", "lra_guard", "lra_warning_semantics", "ab_delta_qc"],
        "complete_when": [
            "lra_loss_and_crest_loss_are_classified_relative_to_input",
            "warnings_preserve_source_vs_mastering_damage_distinction",
            "final_report_surfaces_accept_with_caution_instead_of_masking_damage",
        ],
    },
    {
        "id": "mastering_ownership_dml_contract",
        "title": "Mastering Ownership / DML Final Push Contract",
        "target_patch": "v63.6.2",
        "planned_status": "implemented",
        "source_basis": ["engine_expansion_report", "final_limiter_upgrade_report", "bamix_premaster_rules"],
        "telemetry_keys": ["mastering_ownership", "commercial_push_safety_gate", "v60_deterministic_multistage_limiter", "v6350_limiter_workload_budget_plan", "v6362_dml_ownership_contract"],
        "complete_when": [
            "ownership_class_decides_push_allowed_push_limited_or_artifact_protected_fallback",
            "dml_final_push_is_bounded_by_ownership_budget",
            "target_lufs_miss_is_allowed_when_guarded_by_tp_dynamics_or_artifact_budget",
        ],
        "implemented_pieces": [
            "ownership_decision_and_dml_acceptance_contract",
            "final_limiter_workload_cap_connected_to_dml_parameter_scaling",
            "machine_readable_target_miss_allowed_telemetry",
            "artifact_protected_fallback_blocks_dml_loudness_gain",
            "input_relative_crest_lra_damage_ownership_for_dml",
            "candidate_level_lra_proxy_when_fast_qc_omits_lra",
        ],
        "remaining_work": [
            "proxy_actual_limiter_calibration_and_replay_remain_for_v64",
        ],
    },
    {
        "id": "bamix_premaster_handoff_governance",
        "title": "BAMix Premaster Density / Handoff Governance",
        "target_patch": "v63.0-v63.5",
        "planned_status": "implemented_with_followup",
        "source_basis": ["bamix_premaster_engineering_rules", "stage_m_system_design_report"],
        "telemetry_keys": ["busy_auto_mixing", "busy_auto_mixing_summary", "v6341_reference_db_handoff_contract", "v6350_density_limiter_workload_architecture"],
        "complete_when": [
            "premaster_density_is_created_upstream_not_by_final_limiter_overwork",
            "reference_db_handoff_contract_carries_final_limiter_gain_budget",
            "stem_evidence_does_not_silently_replace_original_stereo_when_bamix_is_unavailable",
        ],
        "remaining_work": [
            "longer_multi_track_validation_of_bamix_commercial_premaster_target_windows",
            "stem_upload_absent_path_should_continue_auto_skip_without_side_effects",
        ],
    },
    {
        "id": "drum_transient_punch",
        "title": "Drum / Transient / Punch Complete DSP",
        "target_patch": "v63.6.1.1",
        "planned_status": "implemented",
        "source_basis": ["quality_upgrade_roadmap", "engine_expansion_report", "stage_m_system_design_report"],
        "telemetry_keys": [
            "v6361_drum_transient_punch",
            "drum_transient_punch",
            "microdynamics_lra_shape_recovery",
            "lra_recovery_stage",
            "mix_conditioning_plan.risk_caps.transient_micro_expander_amount",
        ],
        "implemented_pieces": [
            "microdynamics_lra_shape_recovery_governance",
            "transient_fragility_guard_in_mix_conditioning",
            "final_arbiter_can_reject_louder_candidate_for_crest_or_lra_damage",
            "decay_tail_upward_compression",
            "envelope_transient_reconstruction",
            "drum_punch_body_guard",
            "crest_lra_damage_guard",
            "bass_masking_guard",
            "input_relative_transient_damage_judgment",
            "module_specific_rollback_telemetry",
        ],
        "remaining_work": [],
        "complete_when": [
            "punch_recovery_is_signal_driven_not_profile_exception_driven",
            "crest_lra_damage_guard_measures_pre_post_module_deltas",
            "bass_masking_guard_prevents_kick_snare_recovery_from_worsening_low_end_translation",
            "module_can_bypass_or_rollback_without_hiding_warning_telemetry",
        ],
    },
    {
        "id": "body_center_vocal_support",
        "title": "Body / Center / Vocal Support Complete DSP",
        "target_patch": "v63.7.0",
        "planned_status": "implemented",
        "source_basis": ["stereo_mix_conditioning_report", "engine_expansion_report", "stage_m_system_design_report", "bamix_premaster_rules"],
        "telemetry_keys": ["v6370_body_center_vocal_support"],
        "implemented_pieces": [
            "low_mid_decongestion_proxy",
            "vocal_recession_detection_proxy_through_pqs_and_role_contract",
            "mid_presence_restore_proxy_in_existing_render_chain",
            "dynamic_harmonic_body_fill",
            "vocal_fundamental_ducking",
            "mid_side_low_mid_separation",
            "mud_rollback",
            "center_anchor_or_vocal_hollow_protection",
            "vocal_support_body_layer",
            "guarded_body_center_vocal_telemetry",
        ],
        "remaining_work": [],
        "complete_when": [
            "low_mid_body_is_added_only_when_deficit_not_bottleneck_is_proven",
            "vocal_fundamental_lane_has_priority_over_generic_body_fill",
            "mud_growth_rollback_is_measured_and_reported",
        ],
    },
    {
        "id": "bass_harmonic_translation",
        "title": "Bass Harmonic Translation Complete DSP",
        "target_patch": "v63.8.0",
        "planned_status": "implemented",
        "source_basis": ["engine_expansion_report", "bamix_premaster_rules", "stage_m_system_design_report"],
        "telemetry_keys": ["v6380_bass_harmonic_translation", "bass_harmonic_translation"],
        "implemented_pieces": [
            "translation_aware_low_end_governor",
            "low_side_mono_and_low_end_width_controls",
            "bass_headroom_cleanup_routing_in_research_planner",
            "chebyshev_second_third_harmonic_generation",
            "dc_blocker_after_harmonic_stage",
            "strict_mono_anchor",
            "vocal_low_mid_conflict_notch",
            "harmonic_ratio_telemetry",
        ],
        "remaining_work": [],
        "complete_when": [
            "bass_audibility_improves_on_small_speakers_without_sub_boost",
            "harmonic_stage_cannot_raise_dc_or_low_side_phase_risk",
            "vocal_low_mid_conflict_rolls_back_harmonic_drive",
        ],
    },
    {
        "id": "side_texture_stereo_cleanliness",
        "title": "Side Texture / Stereo Cleanliness Complete DSP",
        "target_patch": "v63.9.0",
        "planned_status": "implemented",
        "source_basis": ["quality_upgrade_roadmap", "stereo_mix_conditioning_report", "ai_residue_research_report", "ssrw_design_report"],
        "telemetry_keys": ["v6390_side_texture_stereo_cleanliness", "side_texture_control", "side_high_hash_fizz_suppressor", "stereo_cleanliness_guard", "ambience_collapse_detection", "mono_fold_down_rollback"],
        "implemented_pieces": [
            "side_texture_control_final_form",
            "side_high_hash_fizz_suppressor_final_form",
            "side_only_band_limited_dynamic_eq_not_broad_high_shelf",
            "stereo_cleanliness_guard_with_center_vocal_brightness_protection",
            "ambience_collapse_detection",
            "mono_fold_down_rollback",
            "codec_playback_translation_proxy_guard",
            "guarded_rollback_reason_telemetry",
        ],
        "remaining_work": [],
        "complete_when": [
            "side_hash_reduction_does_not_kill_intended_ambience_or_hats",
            "mono_fold_down_check_can_trigger_rollback",
            "texture_preservation_and_residue_reduction_conflicts_are_declared_in_telemetry",
            "side_cleanup_is_not_done_by_dml_limiter_push_width_collapse_or_broad_high_shelf",
        ],
    },
    {
        "id": "finishing_intelligence",
        "title": "Finishing Intelligence Foundation",
        "target_patch": "v64.0",
        "planned_status": "implemented",
        "source_basis": ["quality_upgrade_roadmap", "dsp_reference_strategy_report", "final_limiter_upgrade_report", "bamix_premaster_rules"],
        "telemetry_keys": [
            "v6400_finishing_intelligence",
            "playback_translation_auditor",
            "codec_stress_test",
            "shadow_master_candidate_selection",
            "input_relative_damage_ownership",
            "proxy_actual_calibration",
            "decision_replay",
            "corpus_regression",
            "debug_brief",
        ],
        "implemented_pieces": [
            "playback_translation_auditor_proxy",
            "actual_codec_stress_test_or_honest_proxy_fallback",
            "shadow_master_candidate_selection_foundation",
            "input_relative_damage_ownership",
            "proxy_actual_calibration_foundation",
            "decision_replay",
            "corpus_regression_foundation",
            "final_confidence_and_debug_brief_visibility",
        ],
        "remaining_work": [
            "rendered_shadow_candidate_substitution_requires_future_patch_or_explicit_audio_render",
            "large_corpus_proxy_actual_calibration_remains_for_v64_7",
        ],
        "complete_when": [
            "codec_translation_is_measured_on_actual_encoded_audio_when_ffmpeg_succeeds_or_marked_proxy_when_unavailable",
            "shadow_candidate_selection_is_preservation_and_translation_aware_not_loudness_only",
            "decision_replay_can_explain_why_a_target_lufs_was_not_chased",
            "corpus_regression_snapshot_can_be_diffed_across_patch_versions",
        ],
    },
    {
        "id": "commercial_maximum_density",
        "title": "Commercial Maximum Ownership / Density Push Foundation",
        "target_patch": "v64.0.4",
        "planned_status": "implemented",
        "source_basis": ["genre_mastering_reference_v6/v7", "final_limiter_upgrade_report", "bamix_premaster_rules", "user_commercial_maximum_policy"],
        "telemetry_keys": [
            "v6404_commercial_maximum_density",
            "commercial_maximum_density",
            "commercial_density_candidate_lineage",
            "clipper_limiter_candidate_lineage",
            "maximum_push_policy",
            "warning_advisory_policy",
            "codec_safe_vs_maximum_wav_decision",
        ],
        "implemented_pieces": [
            "commercial_density_candidate",
            "clipper_before_limiter",
            "maximum_true_peak_ownership",
            "warning_advisory_policy",
            "codec_safe_candidate_separated_from_maximum_wav_candidate",
        ],
        "complete_when": [
            "commercial_or_maximum_mode_renders_actual_loud_candidate_audio",
            "target_true_peak_dbtp_is_minus_0_1_for_wav_master",
            "warning_labels_are_preserved_as_advisory_unless_severe_damage",
            "codec_safe_distribution_candidate_does_not_block_maximum_wav_master_by_itself",
        ],
    },
)


_CAPABILITY_SIGNAL_MAP: dict[str, tuple[str, ...]] = {
    "drum_transient_punch": (
        "crest_transient_loss",
        "crest_loss_monitor",
        "lra_loss_relative",
        "mastering_induced_lra_collapse",
        "punch_or_transient_loss_risk",
        "candidate_crest_transient_loss",
        "transient_fragility_guard",
    ),
    "body_center_vocal_support": (
        "low_mid_bottleneck",
        "low_mid_decongestion",
        "weak_vocal_anchor",
        "vocal_or_lead_recession_risk",
        "vocal_frontness",
        "vocal_masking",
        "center_hollow",
    ),
    "bass_harmonic_translation": (
        "low_end_headroom_cost",
        "bass_kick_collision",
        "weak_low_punch",
        "low_end_side_excess",
        "low_side_delta_excess",
        "bass_translation_risk",
        "low_end_instability",
    ),
    "side_texture_stereo_cleanliness": (
        "side_high_fizz",
        "side_high_shimmer",
        "side_air_delta_excess",
        "perceptual_harshness_or_ai_hash_risk",
        "phasey_high_side",
        "mono_collapse_risk",
        "stereo_or_mono_translation_risk",
        "artifact_bleed_risk",
    ),
    "mastering_ownership_dml_contract": (
        "reference_limiter_workload_budget_excess",
        "true_peak_room_limited",
        "input_relative_crest_loss_limit",
        "target_lufs_miss_allowed",
        "target_miss_allowed",
        "v6362_dml_ownership_contract",
        "dml_ownership_contract",
        "v60_deterministic_multistage_limiter",
        "dml_contract_route",
        "v6362_dml_candidate_exceeds_final_push_budget",
        "v6362_artifact_protected_fallback_blocks_dml_loudness_gain",
        "v6362_dml_added_lra_loss_exceeds_contract",
        "v6362_dml_added_crest_loss_exceeds_contract",
        "v6362_dml_candidate_exceeds_locked_tp_ceiling",
        "final_limiter_workload_cap_exceeded_upstream_density_required",
        "push_limited",
        "artifact_protected_fallback",
    ),
    "finishing_intelligence": (
        "adaptive_qc_accept_with_caution",
        "codec_risk",
        "codec_induced_damage",
        "translation_risk",
        "playback_translation_auditor",
        "shadow_master_candidate_selection",
        "input_relative_damage_ownership",
        "decision_replay",
        "corpus_regression",
        "final_confidence",
        "proxy_actual_calibration",
    ),
    "commercial_maximum_density": (
        "commercial_maximum_under_pushed",
        "under_pushed",
        "true_peak_room_unused",
        "maximum_true_peak_ownership",
        "commercial_density_candidate",
        "clipper_before_limiter",
        "warning_advisory_policy",
        "codec_safe_vs_maximum_wav_decision",
        "v6404_commercial_maximum_density",
    ),
}


def _truthy(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    return default


def _num(value: Any, default: float | None = 0.0) -> float | None:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _get_path(obj: Any, path: tuple[str, ...], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _has_nested(obj: Any, key: str) -> bool:
    if not isinstance(obj, dict):
        return False
    if key in obj and obj.get(key) not in ({}, [], None):
        return True
    for value in obj.values():
        if isinstance(value, dict) and _has_nested(value, key):
            return True
        if isinstance(value, list):
            for item in value[:80]:
                if isinstance(item, dict) and _has_nested(item, key):
                    return True
    return False


def _compact_value(value: Any) -> Any:
    if isinstance(value, float):
        return round(float(value), 4)
    if isinstance(value, (int, bool)) or value is None:
        return value
    if isinstance(value, str):
        return value[:220]
    if isinstance(value, list):
        return [_compact_value(v) for v in value[:12]]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in list(value.items())[:18]:
            if isinstance(v, (dict, list)):
                out[str(k)] = _compact_value(v)
            else:
                out[str(k)] = _compact_value(v)
        return out
    return str(value)[:180]


def _merge_sources(*sources: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for src in sources:
        if isinstance(src, dict):
            out.update(src)
    return out


def _collect_signal_codes(*sources: Any, limit: int = 600) -> list[str]:
    known_terms = sorted(set(sum((list(v) for v in _CAPABILITY_SIGNAL_MAP.values()), [])))
    found: list[str] = []
    seen: set[str] = set()

    def add(code: Any) -> None:
        text = str(code or "").strip()
        if not text:
            return
        lowered = text.lower()
        for term in known_terms:
            if term in lowered and term not in seen:
                seen.add(term)
                found.append(term)
        if lowered in known_terms and lowered not in seen:
            seen.add(lowered)
            found.append(lowered)

    def walk(obj: Any, depth: int = 0) -> None:
        if len(found) >= limit or depth > 8:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                add(k)
                if str(k).lower() in {"warnings", "warning", "reason", "reasons", "reason_codes", "soft_limit_reasons", "hard_stop_reasons", "selected_blockers_remaining", "blockers_detected", "display_warnings"}:
                    walk(v, depth + 1)
                elif isinstance(v, (dict, list)):
                    walk(v, depth + 1)
                else:
                    add(v)
        elif isinstance(obj, list):
            for item in obj[:120]:
                walk(item, depth + 1)
        elif isinstance(obj, str):
            add(obj)

    for src in sources:
        walk(src)
    return found[:limit]


def _stage_presence(source: dict[str, Any], key: str) -> bool:
    if key in source and source.get(key) not in ({}, [], None):
        return True
    if "." in key:
        parts = tuple(key.split("."))
        return _get_path(source, parts) not in ({}, [], None)
    return _has_nested(source, key)


def _status_from_presence(planned: str, present_count: int, total_count: int) -> str:
    if planned == "implemented":
        return "implemented" if present_count >= max(1, min(total_count, 2)) else "expected_but_not_visible_in_this_run"
    if planned == "implemented_with_followup":
        if present_count >= max(1, min(total_count, 2)):
            return "implemented_with_followup"
        return "partial_or_not_visible_in_this_run"
    if planned == "partial":
        if present_count > 0:
            return "partial"
        return "not_implemented_or_not_visible_in_this_run"
    return planned


def _capability_matrix(source: dict[str, Any], signal_codes: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in _CAPABILITY_REGISTRY:
        cap_id = str(spec.get("id") or "")
        keys = list(spec.get("telemetry_keys") or [])
        # Self-row guard: the framework is being built before its own telemetry
        # keys are attached to the surrounding state/report, so a plain presence
        # scan would incorrectly mark v63.6.0 itself as missing.  If the module
        # reached matrix construction, the framework capability is implemented.
        if cap_id == "research_capability_framework":
            present = list(keys)
            missing: list[str] = []
        else:
            present = [k for k in keys if _stage_presence(source, k)]
            missing = [k for k in keys if k not in present]
        signal_terms = list(_CAPABILITY_SIGNAL_MAP.get(cap_id, ()))
        matched = [c for c in signal_codes if c in signal_terms]
        row = {
            "capability_id": spec.get("id"),
            "title": spec.get("title"),
            "target_patch": spec.get("target_patch"),
            "status": _status_from_presence(str(spec.get("planned_status") or "planned"), len(present), len(keys)),
            "telemetry_present": present[:12],
            "telemetry_missing_or_not_run": missing[:12],
            "source_basis": list(spec.get("source_basis") or []),
            "implemented_pieces": list(spec.get("implemented_pieces") or []),
            "remaining_work": list(spec.get("remaining_work") or []),
            "completion_criteria": list(spec.get("complete_when") or []),
            "matched_current_signals": matched[:12],
            "current_pressure_score": round(float(min(1.0, len(set(matched)) / max(1, len(signal_terms) * 0.35))), 4) if signal_terms else 0.0,
        }
        rows.append(row)
    return rows


def _capability_contracts() -> dict[str, Any]:
    common_guards = [
        "ownership_final_push_budget_is_hard_cap",
        "true_peak_ceiling_must_not_be_raised_by_any_capability",
        "input_relative_lra_and_crest_damage_can_rollback_or_bypass",
        "artifact_risk_can_downscale_or_disable_harmonic_saturation_and_side_processing",
        "module_rollback_must_preserve_warning_telemetry_not_delete_labels",
    ]
    return {
        "schema_version": "busy_research_capability_contracts_v8_5_3_64_0_3",
        "common_input_contract": {
            "required_when_available": [
                "input_analysis",
                "decision.reference_v7_or_genre_chain",
                "mix_conditioning_plan",
                "blocker_remediation_plan",
                "busy_auto_mixing_handoff_policy",
                "mastering_ownership",
                "v60_deterministic_multistage_limiter",
                "adaptive_input_relative_qc_governance",
            ],
            "policy": "Capabilities consume measured evidence and prior telemetry only; they cannot infer permission to push louder from target LUFS alone.",
        },
        "common_output_contract": {
            "required_fields_per_capability": ["enabled", "active", "router_decision", "guards", "rollback", "pre_metrics", "post_metrics", "accepted", "reject_or_bypass_reasons"],
            "telemetry_namespaces": [
                "capability_router",
                "capability_guard_ledger",
                "capability_rollback_ledger",
                "remaining_capability_matrix",
            ],
        },
        "common_guards": common_guards,
        "capabilities": {
            "drum_transient_punch": {
                "inputs": ["crest_factor_db", "lra_lu", "onset_or_transient_proxy", "bass_masking_proxy", "ownership.push_authority"],
                "outputs": ["drum_punch_recovery", "decay_tail_upward_compression", "envelope_transient_reconstruction", "rollback_ledger"],
                "guards": ["max_crest_loss_db", "max_lra_loss_lu", "bass_masking_guard", "tp_room_guard"],
                "rollback_metrics": ["crest_delta_db", "lra_delta_lu", "low_end_masking_delta", "true_peak_delta_db"],
            },
            "body_center_vocal_support": {
                "inputs": ["low_mid_bottleneck", "mud_score", "vocal_front_focus", "vocal_fundamental_proxy", "mid_side_low_mid_balance"],
                "outputs": ["dynamic_harmonic_body_fill", "vocal_fundamental_ducking", "center_anchor_protection", "vocal_support_body_layer", "mid_side_low_mid_separation"],
                "guards": ["mud_growth_guard", "vocal_priority_guard", "low_mid_tp_pressure_guard", "center_side_lowmid_guard", "bus_peak_guard"],
                "rollback_metrics": ["mud_delta", "vocal_prominence_delta", "tp_pressure_delta", "mono_low_mid_delta", "side_lowmid_trim_db", "body_support_assist_rms_db"],
            },
            "bass_harmonic_translation": {
                "inputs": ["low_end_instability", "bass_audibility_proxy", "low_side_ratio", "vocal_low_mid_conflict_proxy"],
                "outputs": ["second_harmonic_ratio", "third_harmonic_ratio", "dc_blocker", "mono_anchor"],
                "guards": ["dc_offset_guard", "low_side_phase_guard", "vocal_low_mid_conflict_notch"],
                "rollback_metrics": ["dc_offset_delta", "low_side_delta", "harmonic_ratio_delta", "vocal_masking_delta"],
            },
            "side_texture_stereo_cleanliness": {
                "status": "implemented",
                "side_texture_control": {"status": "implemented"},
                "stereo_cleanliness": {"status": "implemented"},
                "side_high_hash_fizz_suppressor": {"status": "implemented"},
                "ambience_collapse_detection": {"status": "implemented"},
                "mono_fold_down_rollback": {"status": "implemented"},
                "inputs": ["side_ratio_8k_14k", "artifact_score", "texture_preservation_bias", "mono_fold_down_proxy", "side_high_hash_risk"],
                "outputs": ["side_high_hash_suppressor", "stereo_cleanliness_guard", "ambience_collapse_detection", "mono_fold_down_rollback"],
                "guards": ["intentional_texture_preservation", "center_vocal_brightness_protection", "ambience_collapse_guard", "mono_fold_down_guard", "codec_playback_translation_proxy_guard"],
                "rollback_metrics": ["side_high_delta", "air_delta_db", "correlation_delta", "mono_fold_down_score_delta", "side_width_ratio"],
            },
            "finishing_intelligence": {
                "status": "implemented",
                "playback_translation_auditor": {"status": "implemented"},
                "actual_codec_stress_test": {"status": "implemented_or_honest_proxy_fallback"},
                "shadow_master_candidate_selection": {"status": "implemented_foundation"},
                "input_relative_damage_ownership": {"status": "implemented"},
                "proxy_actual_calibration": {"status": "foundation_implemented"},
                "decision_replay": {"status": "implemented"},
                "corpus_regression": {"status": "foundation_implemented"},
                "inputs": ["delivered_final_audio_metrics", "dml_contract", "v63_6_to_v63_9_module_telemetry", "codec_roundtrip_or_proxy", "stage_state"],
                "outputs": ["translation_risk", "codec_risk", "selected_candidate", "damage_ownership", "proxy_actual_delta", "decision_path", "regression_snapshot"],
                "guards": ["no_warning_hiding", "no_target_chasing", "no_proxy_as_actual", "no_unrendered_shadow_audio_substitution", "preserve_dml_authority"],
                "rollback_metrics": ["codec_risk_delta", "translation_risk", "mastering_lra_delta", "mastering_crest_delta", "selected_candidate_reason"],
            },
            "commercial_maximum_density": {
                "status": "implemented",
                "commercial_maximum_density": {"status": "implemented"},
                "clipper_before_limiter": {"status": "implemented"},
                "commercial_density_candidate": {"status": "implemented"},
                "warning_advisory_policy": {"status": "implemented"},
                "maximum_true_peak_ownership": {"status": "implemented"},
                "inputs": ["current_final_true_peak", "mode_policy", "commercial_finish_report", "dml_contract", "codec_stress_test"],
                "outputs": ["main_clean_candidate", "commercial_density_candidate", "clipper_limiter_candidate", "codec_safe_loud_candidate", "maximum_push_candidate", "selected_candidate_id"],
                "guards": ["no_warning_hiding", "no_proxy_as_actual", "true_peak_below_zero", "severe_distortion_or_collapse_block_only", "codec_safe_separate_from_wav_maximum"],
                "rollback_metrics": ["true_peak_dbtp", "lufs_delta_db", "lra_delta_lu", "crest_delta_db", "blocking_reasons"],
            },
            "mastering_ownership_dml_contract": {
                "inputs": ["bamix_handoff_policy", "target_lufs", "current_lufs", "limiter_workload_budget", "weighted_risk", "tp_room", "standard_lra_lu", "candidate_lra_lu"],
                "outputs": ["selected_loudness_class", "push_authority", "final_push_budget_db", "target_miss_allowed", "contract_passed"],
                "guards": ["limiter_workload_budget", "tp_room", "input_relative_damage", "artifact_protected_fallback", "dml_added_damage_only"],
                "rollback_metrics": ["dml_lufs_delta", "dml_crest_delta", "dml_lra_delta", "workload_excess_lu", "target_gap_after_dml_lu"],
            },
        },
    }


def _router(signal_codes: list[str], matrix: list[dict[str, Any]], source: dict[str, Any]) -> dict[str, Any]:
    route_entries: list[dict[str, Any]] = []
    by_id = {str(r.get("capability_id")): r for r in matrix}
    for cap_id, terms in _CAPABILITY_SIGNAL_MAP.items():
        matched = [c for c in signal_codes if c in terms]
        row = by_id.get(cap_id, {})
        if matched or str(row.get("status")) in {"partial", "implemented_with_followup"}:
            status = str(row.get("status") or "")
            if cap_id == "drum_transient_punch" and matched:
                next_action = "invoke_v6361_guarded_capability_or_record_bypass"
            elif cap_id == "mastering_ownership_dml_contract" and matched:
                next_action = "use_v6362_ownership_dml_contract_or_record_target_miss_policy"
            elif cap_id == "body_center_vocal_support" and matched and status == "implemented":
                next_action = "use_v6370_body_center_vocal_support_or_record_guarded_bypass"
            elif cap_id == "bass_harmonic_translation" and matched and status == "implemented":
                next_action = "use_v6380_bass_harmonic_translation_or_record_guarded_bypass"
            elif cap_id == "side_texture_stereo_cleanliness" and matched and status == "implemented":
                next_action = "use_v6390_side_texture_stereo_cleanliness_or_record_guarded_bypass"
            elif cap_id == "commercial_maximum_density" and status == "implemented":
                next_action = "use_v6404_commercial_maximum_density_or_record_warning_advisory_bypass"
            elif status in {"partial", "not_implemented_or_not_visible_in_this_run"} and matched:
                next_action = "implement_complete_capability_patch"
            else:
                next_action = "monitor_or_use_existing_guard"
            route_entries.append({
                "capability_id": cap_id,
                "matched_signals": matched[:12],
                "status": row.get("status"),
                "next_action": next_action,
                "activation_mode": "capability_plus_router_guard_rollback_telemetry",
                "forbidden_mode": "single_label_hotfix_or_always_on_dsp",
                "priority_score": round(float(row.get("current_pressure_score") or 0.0), 4),
            })
    route_entries.sort(key=lambda x: (float(x.get("priority_score") or 0.0), bool(x.get("matched_signals"))), reverse=True)
    ownership = source.get("mastering_ownership") if isinstance(source.get("mastering_ownership"), dict) else {}
    v6350_plan = source.get("v6350_limiter_workload_budget_plan") if isinstance(source.get("v6350_limiter_workload_budget_plan"), dict) else {}
    if not ownership and isinstance(v6350_plan.get("mastering_ownership"), dict):
        ownership = v6350_plan.get("mastering_ownership") or {}
    dml = source.get("v60_deterministic_multistage_limiter") if isinstance(source.get("v60_deterministic_multistage_limiter"), dict) else {}
    dml_contract = source.get("v6362_dml_ownership_contract") if isinstance(source.get("v6362_dml_ownership_contract"), dict) else (dml.get("v6362_dml_ownership_contract") if isinstance(dml.get("v6362_dml_ownership_contract"), dict) else {})
    return {
        "schema_version": "busy_capability_router_v8_5_3_64_0_3",
        "active": True,
        "router_mode": "research_capability_matrix_to_feature_patch_queue",
        "route_entries": route_entries[:16],
        "current_result_signal_codes": signal_codes[:80],
        "ownership_context": {
            "selected_loudness_class": ownership.get("selected_loudness_class"),
            "push_authority": ownership.get("push_authority"),
            "push_allowed": ownership.get("push_allowed"),
            "push_limited": ownership.get("push_limited"),
            "final_push_budget_db": ownership.get("final_push_budget_db"),
            "limiter_workload_budget_lu": ownership.get("limiter_workload_budget_lu"),
        },
        "dml_context": {
            "active": dml.get("active"),
            "accepted": dml.get("accepted"),
            "applied": dml.get("applied"),
            "role": dml.get("role"),
            "reason": dml.get("reason"),
            "contract_active": dml_contract.get("active"),
            "target_miss_allowed": dml_contract.get("target_miss_allowed"),
            "contract_passed": dml_contract.get("contract_passed"),
        },
        "policy": "Route recurring blockers/warnings to complete capability patches. Do not create a one-off hotfix for a warning label.",
    }


def _conflict_policy() -> dict[str, Any]:
    return {
        "schema_version": "busy_research_capability_conflict_policy_v8_5_3_64_0_3",
        "active": True,
        "hard_rules": [
            {
                "id": "ownership_over_target_lufs",
                "rule": "target_lufs_is_never_permission_to_exceed_mastering_ownership.final_push_budget_db_or_tp_ceiling",
            },
            {
                "id": "bamix_density_before_dml",
                "rule": "BAMix/premaster density must reduce final limiter workload; DML cannot be used as a rescue-gain stage when handoff budget is exceeded.",
            },
            {
                "id": "v6362_dml_target_miss_contract",
                "rule": "DML may accept a quieter-than-target final only when target_miss_allowed is explicitly owned by limiter workload, TP room, artifact fallback, or input-relative damage budget.",
            },
            {
                "id": "body_fill_vs_low_mid_bottleneck",
                "rule": "Body fill is disabled or rolled back when low_mid_bottleneck/mud/tp_pressure worsens; body deficit must be proven separately from mud.",
            },
            {
                "id": "bass_harmonics_vs_vocal_fundamental",
                "rule": "Bass harmonic translation must notch or downscale when vocal low-mid/fundamental conflict increases.",
            },
            {
                "id": "side_cleanup_vs_texture",
                "rule": "Side hash suppression cannot collapse intentional ambience, hats, vocal air, cowbell, grit or lo-fi texture; texture gate wins until residue confidence is high.",
            },
            {
                "id": "transient_recovery_vs_tp_room",
                "rule": "Transient reconstruction must reserve TP room and roll back if it forces extra limiting or increases crest/LRA damage after final safety.",
            },
        ],
        "policy": "Conflicts are handled by guard/rollback ledgers, not hidden exceptions. This is framework-routed capability telemetry in v63.6.2; only explicitly implemented modules may render audio under guard.",
    }


def _summary(matrix: list[dict[str, Any]], router: dict[str, Any]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in matrix:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    high_pressure = [r for r in (router.get("route_entries") or []) if isinstance(r, dict) and r.get("matched_signals")]
    return {
        "schema_version": "busy_research_capability_summary_v8_5_3_64_0_3",
        "status_counts": counts,
        "highest_pressure_capabilities": [
            {
                "capability_id": r.get("capability_id"),
                "priority_score": r.get("priority_score"),
                "matched_signals": r.get("matched_signals"),
                "next_action": r.get("next_action"),
            }
            for r in high_pressure[:6]
        ],
        "next_patch_queue": [
            "v64.1_perceptual_quality_score_listener_fatigue_scoring",
            "v64.2_loudness_feasibility_predictor_candidate_pruning",
            "v64.3_section_aware_stress_window_sentinel_planner",
        ],
    }


def build_research_capability_framework(
    features: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    report: dict[str, Any] | None = None,
    *,
    stage: str = "planning",
) -> dict[str, Any]:
    """Build the v64.0 research capability matrix and router contract."""
    _enabled_raw = os.environ.get("BUSY_V6362_RESEARCH_CAPABILITY_FRAMEWORK", os.environ.get("BUSY_V6361_RESEARCH_CAPABILITY_FRAMEWORK", os.environ.get("BUSY_V6360_RESEARCH_CAPABILITY_FRAMEWORK", "1")))
    enabled = _truthy(_enabled_raw, True)
    base: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "engine_version": ENGINE_VERSION,
        "enabled": bool(enabled),
        "active": False,
        "stage": stage,
        "mode": "router_contract_plus_guarded_capability_dsp",
    }
    if not enabled:
        base["reason"] = "BUSY_V6362_RESEARCH_CAPABILITY_FRAMEWORK disabled"
        return base

    features = features if isinstance(features, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    report = report if isinstance(report, dict) else {}
    # Later sources override earlier ones for top-level direct lookups while
    # recursive signal scanning still sees all three payloads.
    source = _merge_sources(features, decision, report)
    # Commonly nested final report/decision fields are elevated for status checks.
    if isinstance(report.get("output_analysis"), dict):
        source.setdefault("output_analysis", report.get("output_analysis"))
    if isinstance(decision.get("final_decision_context"), dict):
        source.update({k: v for k, v in decision.get("final_decision_context", {}).items() if k not in source})

    signal_codes = _collect_signal_codes(features, decision, report)
    matrix = _capability_matrix(source, signal_codes)
    contracts = _capability_contracts()
    router = _router(signal_codes, matrix, source)
    conflict = _conflict_policy()
    summary = _summary(matrix, router)

    base.update({
        "active": True,
        "remaining_capability_matrix": matrix,
        "capability_router": router,
        "capability_contracts": contracts,
        "capability_conflict_policy": conflict,
        "summary": summary,
        "handoff_order": [
            "analysis_and_private_reference_context",
            "pre_clean",
            "bamix_or_original_stereo_handoff",
            "mix_conditioning_and_blocker_planning",
            "v63_6_2_research_capability_framework_router",
            "v63_6_1_1_drum_transient_punch_when_routed_and_guarded",
            "pre_limiter_render",
            "v63_6_2_mastering_ownership_dml_contract",
            "final_limiter_and_qc",
            "v64_0_3_finishing_intelligence_selection_semantics_hotfix",
            "v64_0_4_commercial_maximum_density_push_foundation",
            "debug_brief_and_corpus_regression_snapshot",
        ],
        "implementation_policy": {
            "v63_6_1_1_dsp_behavior": "drum_transient_punch_complete_dsp_may_render_audio_only_when_routed_eligible_and_guard_accepted",
            "v63_6_2_dml_contract_behavior": "dml_may_polish_or_catch_peaks_only_inside_mastering_ownership_and_final_limiter_workload_budget",
            "framework_non_dsp_policy": "framework_does_not_relax_qc_thresholds_hide_warnings_or_push_lufs",
            "v64_0_3_finishing_intelligence_behavior": "report_only_playback_codec_shadow_damage_replay_regression_foundation_without_lufs_push_or_warning_hiding",
            "v64_0_4_commercial_maximum_density_behavior": "actual_rendered_commercial_maximum_wav_density_candidates_with_warning_advisory_policy_and_severe_damage_blocks_only",
            "future_patch_policy": "complete_capability_implementation_per_patch_with_router_guard_rollback_telemetry",
            "forbidden_policy": "do_not_patch_individual_warning_labels_or_add_always_on_dsp",
            "zip_policy": "patch_zip_contains_modified_files_only",
        },
    })
    return base


def apply_research_capability_framework(
    features: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    *,
    stage: str = "planning",
    report: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Attach framework telemetry to both features and decision.

    Returns (features, decision, framework).  The returned decision is a shallow
    copy of the input so callers can safely rebind it without mutating older
    state snapshots unexpectedly.
    """
    f = features if isinstance(features, dict) else {}
    d = copy.deepcopy(decision or {})
    fw = build_research_capability_framework(f, d, report, stage=stage)
    f["research_capability_framework"] = fw
    f["remaining_capability_matrix"] = fw.get("remaining_capability_matrix", []) if isinstance(fw, dict) else []
    f["capability_router"] = fw.get("capability_router", {}) if isinstance(fw, dict) else {}
    f["capability_contracts"] = fw.get("capability_contracts", {}) if isinstance(fw, dict) else {}
    f["capability_conflict_policy"] = fw.get("capability_conflict_policy", {}) if isinstance(fw, dict) else {}
    d["research_capability_framework"] = fw
    d["remaining_capability_matrix"] = f.get("remaining_capability_matrix", [])
    d["capability_router"] = f.get("capability_router", {})
    d["capability_contracts"] = f.get("capability_contracts", {})
    d["capability_conflict_policy"] = f.get("capability_conflict_policy", {})
    d.setdefault("final_decision_context", {})["research_capability_framework"] = {
        "schema_version": fw.get("schema_version") if isinstance(fw, dict) else SCHEMA_VERSION,
        "active": bool(fw.get("active")) if isinstance(fw, dict) else False,
        "summary": fw.get("summary", {}) if isinstance(fw, dict) else {},
        "policy": "Full capability matrix is stored at decision.research_capability_framework and report.research_capability_framework.",
    }
    flags = list(d.get("processing_flags") or []) if isinstance(d.get("processing_flags"), list) else []
    if isinstance(fw, dict) and bool(fw.get("active")):
        for flag in ["research_capability_framework_v6380", "research_capability_framework_v6370", "research_capability_framework_v6362", "research_capability_framework_v6361", "remaining_capability_matrix_active", "capability_router_contract_active", "drum_transient_punch_capability_available", "dml_ownership_contract_capability_available", "body_center_vocal_support_capability_available", "bass_harmonic_translation_capability_available", "v6380_bass_harmonic_translation_capability_available", "v6390_side_texture_stereo_cleanliness_capability_available", "v6400_finishing_intelligence_capability_available"]:
            if flag not in flags:
                flags.append(flag)
        for inactive_flag in ["research_capability_framework_v6380_inactive", "research_capability_framework_v6370_inactive", "research_capability_framework_v6362_inactive", "research_capability_framework_v6361_inactive"]:
            if inactive_flag in flags:
                flags = [x for x in flags if x != inactive_flag]
    else:
        # Do not advertise active router/matrix flags when the framework is
        # disabled by env or fell back inactive.  This keeps deployment toggles
        # and stage_state smoke tests honest.
        flags = [x for x in flags if x not in {"research_capability_framework_v6380", "research_capability_framework_v6370", "research_capability_framework_v6362", "research_capability_framework_v6361", "remaining_capability_matrix_active", "capability_router_contract_active", "drum_transient_punch_capability_available", "dml_ownership_contract_capability_available", "body_center_vocal_support_capability_available", "bass_harmonic_translation_capability_available", "v6380_bass_harmonic_translation_capability_available", "v6390_side_texture_stereo_cleanliness_capability_available", "v6400_finishing_intelligence_capability_available"}]
        if "research_capability_framework_v6380_inactive" not in flags:
            flags.append("research_capability_framework_v6380_inactive")
    d["processing_flags"] = flags
    return f, d, fw


def attach_research_capability_to_report(
    report: dict[str, Any] | None,
    *,
    features: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    stage: str = "final_report",
) -> dict[str, Any]:
    """Surface framework telemetry in the final user/debug report."""
    out = dict(report or {})
    f = features if isinstance(features, dict) else {}
    d = decision if isinstance(decision, dict) else {}
    fw = out.get("research_capability_framework") if isinstance(out.get("research_capability_framework"), dict) else None
    if not fw:
        fw = d.get("research_capability_framework") if isinstance(d.get("research_capability_framework"), dict) else None
    if not fw:
        fw = f.get("research_capability_framework") if isinstance(f.get("research_capability_framework"), dict) else None
    if not fw or not isinstance(fw, dict) or fw.get("stage") != stage or fw.get("schema_version") != SCHEMA_VERSION:
        fw = build_research_capability_framework(f, d, out, stage=stage)
    out["research_capability_framework"] = fw
    out["remaining_capability_matrix"] = fw.get("remaining_capability_matrix", [])
    out["capability_router"] = fw.get("capability_router", {})
    out["capability_contracts"] = fw.get("capability_contracts", {})
    out["capability_conflict_policy"] = fw.get("capability_conflict_policy", {})
    if fw.get("active"):
        # v63.6.2.11: this helper is a framework/report attachment, not the final
        # delivered-source authority.  If the DML report-authority finalizer has
        # already stamped the report, keep that engine/schema intact and surface
        # framework labels only in framework_* fields.  run_stage_limit and
        # single-stage also re-apply the DML finalizer after this helper, but this
        # local guard prevents future call-order regressions.
        report_authority_active = bool(
            (isinstance(out.get("v6362_10_report_authority_finalizer"), dict) and out["v6362_10_report_authority_finalizer"].get("active"))
            or (isinstance(out.get("v6362_9_report_authority_finalizer"), dict) and out["v6362_9_report_authority_finalizer"].get("active"))
            or (isinstance(out.get("v6362_8_report_authority_finalizer"), dict) and out["v6362_8_report_authority_finalizer"].get("active"))
        )
        if report_authority_active:
            out["framework_engine_version"] = ENGINE_VERSION
            out["framework_mastering_report_schema_version"] = REPORT_SCHEMA
            out["research_capability_framework_attachment_policy_v6362_10"] = {
                "active": True,
                "reason": "preserved_existing_dml_report_authority_engine_schema",
                "policy": "Research Capability Framework attachment must not override delivered DML report-authority engine/schema after the report authority finalizer has run.",
            }
            return out
        # Preserve the actual audio-render/base engine id across repeated report
        # attachment calls.  v63.6.x may attach the matrix at A/B checkpoints and
        # again at final report time; the second call must not overwrite the
        # original v63.5.x base engine with the framework engine id.
        current_engine = out.get("engine_version")
        # v63.6.0.2: stage_state can be passed through this helper before the
        # audio-render report exists.  Do not create null base_* lineage fields
        # in that case; only preserve a real pre-framework engine marker.
        if current_engine and current_engine != ENGINE_VERSION and not out.get("base_engine_version"):
            out["base_engine_version"] = current_engine
        current_schema = out.get("mastering_report_schema_version")
        if current_schema and current_schema != REPORT_SCHEMA and not out.get("base_mastering_report_schema_version"):
            out["base_mastering_report_schema_version"] = current_schema
        out["framework_engine_version"] = ENGINE_VERSION
        out["framework_mastering_report_schema_version"] = REPORT_SCHEMA
        out["engine_version"] = ENGINE_VERSION
        out["mastering_report_schema_version"] = REPORT_SCHEMA
    return out


def compact_research_capability_summary(framework: dict[str, Any] | None) -> dict[str, Any]:
    fw = framework if isinstance(framework, dict) else {}
    return {
        "schema_version": "busy_research_capability_compact_summary_v8_5_3_64_0_3",
        "active": bool(fw.get("active")),
        "summary": _compact_value(fw.get("summary", {})),
        "router": _compact_value(fw.get("capability_router", {})),
    }
