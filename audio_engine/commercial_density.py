from __future__ import annotations

"""v64.0.4 Commercial/Maximum density push foundation.

This module intentionally separates "delivery selection" from codec/translation
caution.  Commercial/Maximum modes must render an actual WAV-density candidate
near the requested -0.1 dBTP ceiling; codec-safe distribution remains a separate
candidate/telemetry path instead of blocking the WAV master push.
"""

from typing import Any, Callable

import numpy as np

from .analyzer import analyze_audio_fast_qc
from .dynamics import apply_gain_db, lookahead_limiter, soft_clip, subtle_saturation, upward_parallel_density
from .oversampling import normalize_to_true_peak_chunked, oversampled_limit_chunked, true_peak_db_oversampled_chunked

ENGINE_VERSION = "busy_auto_mastering_v8_5_3_64_0_4_commercial_maximum_density_push_foundation"
REPORT_SCHEMA = "busy_commercial_maximum_density_v8_5_3_64_0_4"


def _finite_float(value: Any, fallback: float = 0.0) -> float:
    try:
        f = float(value)
        if np.isfinite(f):
            return f
    except Exception:
        pass
    return float(fallback)


def _metric(analysis: dict[str, Any] | None, *names: str, default: float = 0.0) -> float:
    analysis = analysis if isinstance(analysis, dict) else {}
    for name in names:
        if name in analysis:
            return _finite_float(analysis.get(name), default)
    return float(default)


def _summary(analysis: dict[str, Any] | None) -> dict[str, Any]:
    analysis = analysis if isinstance(analysis, dict) else {}
    return {
        "integrated_lufs": round(_metric(analysis, "integrated_lufs", "lufs", default=-99.0), 3),
        "true_peak_dbtp": round(_metric(analysis, "approx_true_peak_dbfs", "true_peak_dbfs", default=-99.0), 3),
        "lra_lu": round(_metric(analysis, "lra_lu", "loudness_range_lu", default=0.0), 3),
        "crest_factor_db": round(_metric(analysis, "crest_factor_db", "crest_db", default=0.0), 3),
        "sample_peak_dbfs": round(_metric(analysis, "sample_peak_dbfs", "peak_dbfs", default=-99.0), 3),
    }


def _mode_policy(decision: dict[str, Any] | None, mode: str | None) -> str:
    d = decision if isinstance(decision, dict) else {}
    policy = d.get("auto_intensity_policy") if isinstance(d.get("auto_intensity_policy"), dict) else {}
    raw = (
        policy.get("final_mode")
        or d.get("mastering_intensity")
        or d.get("selected_intensity")
        or mode
        or "balanced"
    )
    text = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "auto_commercial_master": "commercial",
        "commercial_reference": "commercial",
        "commercial_reference_match": "commercial",
        "commercial_loud": "commercial",
        "max": "maximum",
        "maximum_push": "maximum",
        "loudest": "maximum",
        "hot": "maximum",
        "safe_clean": "safe",
        "streaming": "safe",
        "auto": "commercial" if "commercial" in str(mode or "").lower() else "balanced",
    }
    text = aliases.get(text, text)
    if "maximum" in text:
        return "maximum"
    if "commercial" in text:
        return "commercial"
    if "safe" in text or "clean" in text:
        return "safe"
    return "balanced"


def _analysis_with_true_peak(
    audio: np.ndarray,
    sr: int,
    analyzer: Callable[[np.ndarray, int], dict[str, Any]] | None,
    oversample: int,
) -> dict[str, Any]:
    try:
        analysis = analyzer(audio, sr) if callable(analyzer) else analyze_audio_fast_qc(audio, sr, true_peak_oversample=oversample)
        if not isinstance(analysis, dict):
            analysis = {}
    except Exception:
        analysis = {}
    if "approx_true_peak_dbfs" not in analysis and "true_peak_dbfs" not in analysis:
        try:
            analysis["approx_true_peak_dbfs"] = float(true_peak_db_oversampled_chunked(audio, oversample=max(1, int(oversample)), sr=sr))
        except Exception:
            pass
    return analysis


def _safe_array(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _limit_and_pin_tp(audio: np.ndarray, sr: int, target_tp: float, oversample: int, *, lookahead_ms: float, release_ms: float) -> tuple[np.ndarray, dict[str, Any]]:
    limited = oversampled_limit_chunked(
        audio,
        sr=sr,
        ceiling_db=float(target_tp),
        oversample=max(1, int(oversample)),
        lookahead_ms=float(lookahead_ms),
        release_ms=float(release_ms),
    )
    pinned, gain_db, tp2 = normalize_to_true_peak_chunked(
        limited,
        target_db=float(target_tp),
        oversample=max(1, int(oversample)),
        sr=sr,
    )
    # Absolute safety: never leave an oversampled value above 0 dBTP if the
    # normalizer observed an edge-case overshoot.
    try:
        tp_final = float(true_peak_db_oversampled_chunked(pinned, oversample=max(1, int(oversample)), sr=sr))
        if np.isfinite(tp_final) and tp_final > -0.01:
            corr = -0.05 - tp_final
            pinned = apply_gain_db(pinned, corr)
            gain_db += corr
            tp2 = true_peak_db_oversampled_chunked(pinned, oversample=max(1, int(oversample)), sr=sr)
    except Exception:
        pass
    return np.nan_to_num(pinned.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0), {
        "limiter_ceiling_dbtp": round(float(target_tp), 3),
        "true_peak_pin_gain_db": round(float(gain_db), 3),
        "post_pin_true_peak_dbtp": round(_finite_float(tp2, -99.0), 3),
        "lookahead_ms": round(float(lookahead_ms), 3),
        "release_ms": round(float(release_ms), 3),
    }


def _build_candidate(
    name: str,
    base_audio: np.ndarray,
    sr: int,
    *,
    target_tp: float,
    target_lift_db: float,
    density_drive_amount: float,
    clipper_drive_db: float,
    clipper_mix: float,
    parallel_density_mix: float,
    limiter_gain_db: float,
    oversample: int,
    analyzer: Callable[[np.ndarray, int], dict[str, Any]] | None,
    codec_safe: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = _safe_array(base_audio)
    stage_trace: list[dict[str, Any]] = []

    density_drive_db = max(0.0, float(density_drive_amount) * 1.6)
    if density_drive_db > 0.01:
        x = subtle_saturation(x, drive_db=float(density_drive_db), mix=min(0.42, 0.12 + float(density_drive_amount) * 0.22)).astype(np.float32, copy=False)
        stage_trace.append({"stage": "broadband_subtle_density_drive", "drive_db": round(float(density_drive_db), 3), "amount": round(float(density_drive_amount), 3)})

    if parallel_density_mix > 0.005:
        x = upward_parallel_density(x, sr, threshold_db=-40.0 if not codec_safe else -42.0, ratio=1.35 if codec_safe else 1.55, mix=float(parallel_density_mix)).astype(np.float32, copy=False)
        stage_trace.append({"stage": "parallel_density_compression", "mix": round(float(parallel_density_mix), 3), "codec_safe": bool(codec_safe)})

    pregain = max(0.0, min(float(target_lift_db), float(target_lift_db) * (0.50 if codec_safe else 0.62)))
    if pregain > 0.005:
        x = apply_gain_db(x, pregain).astype(np.float32, copy=False)
        stage_trace.append({"stage": "pre_clipper_makeup_gain", "gain_db": round(float(pregain), 3)})

    if clipper_drive_db > 0.005 and clipper_mix > 0.005:
        x = soft_clip(x, drive_db=float(clipper_drive_db), mix=float(clipper_mix)).astype(np.float32, copy=False)
        stage_trace.append({"stage": "soft_clipper_before_limiter", "drive_db": round(float(clipper_drive_db), 3), "mix": round(float(clipper_mix), 3)})

    post_clip_gain = max(0.0, float(limiter_gain_db))
    if post_clip_gain > 0.005:
        x = apply_gain_db(x, post_clip_gain).astype(np.float32, copy=False)
        stage_trace.append({"stage": "limiter_input_makeup_gain", "gain_db": round(float(post_clip_gain), 3)})

    x = lookahead_limiter(x, sr, ceiling_db=float(target_tp), lookahead_ms=1.4 if not codec_safe else 2.5, release_ms=115.0 if not codec_safe else 150.0).astype(np.float32, copy=False)
    stage_trace.append({"stage": "program_limiter_workload_distribution", "ceiling_dbtp": round(float(target_tp), 3)})

    x, limit_report = _limit_and_pin_tp(x, sr, target_tp, oversample, lookahead_ms=1.0 if not codec_safe else 2.0, release_ms=135.0 if not codec_safe else 180.0)
    analysis = _analysis_with_true_peak(x, sr, analyzer, oversample)
    return x, {
        "candidate_id": name,
        "actual_audio_rendered": True,
        "codec_safe_distribution_candidate": bool(codec_safe),
        "target_true_peak_dbtp": round(float(target_tp), 3),
        "push_gain_db": round(float(target_lift_db), 3),
        "clipper_drive_db": round(float(clipper_drive_db), 3),
        "clipper_mix": round(float(clipper_mix), 3),
        "limiter_gain_db": round(float(limiter_gain_db), 3),
        "density_drive_amount": round(float(density_drive_amount), 3),
        "parallel_density_mix": round(float(parallel_density_mix), 3),
        "stage_trace": stage_trace,
        "limiter": limit_report,
        "metrics": _summary(analysis),
        "analysis": analysis,
    }


def _damage_advisories(base: dict[str, Any], cand: dict[str, Any], mode_policy: str) -> tuple[list[dict[str, Any]], list[str]]:
    advisories: list[dict[str, Any]] = []
    blocking: list[str] = []
    base_m = _summary(base)
    cand_m = _summary(cand)
    lra_delta = float(cand_m.get("lra_lu", 0.0) or 0.0) - float(base_m.get("lra_lu", 0.0) or 0.0)
    crest_delta = float(cand_m.get("crest_factor_db", 0.0) or 0.0) - float(base_m.get("crest_factor_db", 0.0) or 0.0)
    tp = float(cand_m.get("true_peak_dbtp", -99.0) or -99.0)
    lufs_delta = float(cand_m.get("integrated_lufs", -99.0) or -99.0) - float(base_m.get("integrated_lufs", -99.0) or -99.0)

    def add(name: str, value: float, threshold: float, severe: bool = False) -> None:
        advisory = {
            "name": name,
            "value": round(float(value), 3),
            "threshold": round(float(threshold), 3),
            "mode": mode_policy,
            "advisory": True,
            "blocking": bool(severe),
        }
        advisories.append(advisory)
        if severe:
            blocking.append(name)

    if lra_delta < -1.20:
        add("lra_loss_relative", lra_delta, -1.20, severe=bool(mode_policy in {"safe", "balanced"} or lra_delta < -2.40))
    if crest_delta < -2.25:
        add("crest_loss_relative", crest_delta, -2.25, severe=bool(mode_policy in {"safe", "balanced"} or crest_delta < -4.20))
    if lufs_delta > 3.8 and float(cand_m.get("crest_factor_db", 99.0) or 99.0) < 3.8:
        add("obvious_over_flattening", lufs_delta, 3.8, severe=True)
    if tp > -0.01:
        add("true_peak_overshoot", tp, -0.01, severe=True)
    if not np.isfinite(tp) or not np.isfinite(float(cand_m.get("integrated_lufs", -99.0) or -99.0)):
        blocking.append("non_finite_candidate_metrics")
    return advisories, sorted(set(blocking))


def _candidate_score(meta: dict[str, Any], base_analysis: dict[str, Any], target_tp: float, mode_policy: str) -> float:
    metrics = meta.get("metrics") if isinstance(meta.get("metrics"), dict) else {}
    base = _summary(base_analysis)
    lufs = float(metrics.get("integrated_lufs", -99.0) or -99.0)
    base_lufs = float(base.get("integrated_lufs", -99.0) or -99.0)
    tp = float(metrics.get("true_peak_dbtp", -99.0) or -99.0)
    crest = float(metrics.get("crest_factor_db", 0.0) or 0.0)
    lra = float(metrics.get("lra_lu", 0.0) or 0.0)
    target_error = abs(float(target_tp) - tp)
    score = 0.0
    score += max(0.0, lufs - base_lufs) * (10.0 if mode_policy == "commercial" else 12.0)
    score += max(0.0, 2.0 - target_error * 10.0) * (10.0 if mode_policy == "maximum" else 7.0)
    score += 12.0 if -0.30 <= tp <= -0.05 else -12.0 if tp <= -1.0 else 0.0
    score += min(8.0, max(0.0, crest - 3.8) * 1.1)
    score += min(4.0, max(0.0, lra - 0.9) * 1.4)
    if meta.get("candidate_id") == "maximum_push_candidate" and mode_policy == "maximum":
        score += 8.0
    if meta.get("candidate_id") == "commercial_density_candidate" and mode_policy == "commercial":
        score += 5.0
    if meta.get("codec_safe_distribution_candidate") and mode_policy in {"commercial", "maximum"}:
        score -= 7.0
    return round(float(score), 3)


def apply_commercial_maximum_density_push(
    y: np.ndarray,
    sr: int,
    *,
    before_analysis: dict[str, Any] | None = None,
    current_analysis: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    mode: str | None = None,
    commercial_finish_report: dict[str, Any] | None = None,
    true_peak_ceiling_db: float = -0.1,
    oversample: int = 4,
    analyzer: Callable[[np.ndarray, int], dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Render and optionally select an actual Commercial/Maximum density candidate.

    Safe/Balanced do not change the delivered buffer.  Commercial/Maximum render
    all required candidate classes and select a loud WAV-master candidate unless
    a severe hard block is measured.
    """
    policy = _mode_policy(decision, mode)
    enabled = str((decision or {}).get("disable_v6404_commercial_maximum_density", "")).lower() not in {"1", "true", "yes", "on"}
    x = _safe_array(y)
    base_analysis = current_analysis if isinstance(current_analysis, dict) and current_analysis else _analysis_with_true_peak(x, sr, analyzer, oversample)
    base_metrics = _summary(base_analysis)

    env_target = _finite_float((decision or {}).get("v6404_target_true_peak_dbtp"), _finite_float(true_peak_ceiling_db, -0.1))
    if policy in {"commercial", "maximum"}:
        target_tp = float(np.clip(env_target if env_target > -0.35 else -0.1, -0.30, -0.05))
    else:
        target_tp = float(np.clip(_finite_float(true_peak_ceiling_db, -1.0), -2.5, -0.05))

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "engine_version": ENGINE_VERSION,
        "enabled": bool(enabled),
        "active": bool(enabled and policy in {"commercial", "maximum"}),
        "applied": False,
        "mode_policy": policy,
        "target_true_peak_dbtp": round(float(target_tp), 3),
        "input_metrics": base_metrics,
        "final_true_peak_dbtp": base_metrics.get("true_peak_dbtp"),
        "under_pushed": bool(policy in {"commercial", "maximum"} and float(base_metrics.get("true_peak_dbtp", -99.0) or -99.0) <= -1.0),
        "candidate_policy": "render actual loud WAV candidates; codec-safe distribution is separate telemetry and does not block Commercial/Maximum WAV master unless severe destruction is measured",
        "warning_policy": {
            "safe": "warnings may block",
            "balanced": "warnings can block when damage exceeds budget",
            "commercial": "warnings are advisory except severe clipping/distortion/collapse",
            "maximum": "warnings are advisory except severe clipping/distortion/collapse; -0.1 dBTP WAV target is honored",
        },
        "candidates": [],
        "damage_advisories": [],
        "blocking_reasons": [],
    }
    if not enabled:
        report["reason"] = "disabled_by_decision"
        return x, report
    if policy not in {"commercial", "maximum"}:
        report["reason"] = "mode_policy_does_not_select_loud_candidate"
        report["warning_blocking"] = True
        report["warning_advisory"] = False
        return x, report

    base_tp = float(base_metrics.get("true_peak_dbtp", -99.0) or -99.0)
    tp_room_lift = max(0.0, float(target_tp) - base_tp)
    finish = commercial_finish_report if isinstance(commercial_finish_report, dict) else {}
    target_lufs = _finite_float(finish.get("target_lufs"), _finite_float(((decision or {}).get("targets") or {}).get("integrated_lufs_target") if isinstance((decision or {}).get("targets"), dict) else None, -8.8))
    base_lufs = float(base_metrics.get("integrated_lufs", -99.0) or -99.0)
    lufs_gap = max(0.0, target_lufs - base_lufs)
    # Push is driven by both unused TP headroom and target gap.  It is capped,
    # but never zero for an under-pushed commercial/maximum master.
    desired_lift = max(tp_room_lift, min(4.2, lufs_gap * 0.85))
    if report["under_pushed"]:
        desired_lift = max(desired_lift, min(2.2, tp_room_lift + 0.20))
    desired_lift = float(np.clip(desired_lift, 0.15, 5.2 if policy == "maximum" else 4.2))

    # Required actual candidate classes.
    main_meta = {
        "candidate_id": "main_clean_candidate",
        "actual_audio_rendered": True,
        "selected_delivery_candidate": False,
        "target_true_peak_dbtp": round(float(target_tp), 3),
        "push_gain_db": 0.0,
        "clipper_drive_db": 0.0,
        "limiter_gain_db": 0.0,
        "density_drive_amount": 0.0,
        "parallel_density_mix": 0.0,
        "metrics": base_metrics,
        "analysis": base_analysis,
    }
    candidates: list[tuple[str, np.ndarray, dict[str, Any]]] = [("main_clean_candidate", x, main_meta)]

    specs = [
        (
            "commercial_density_candidate",
            dict(
                target_tp=target_tp,
                target_lift_db=desired_lift * 0.86,
                density_drive_amount=0.28 if policy == "commercial" else 0.34,
                clipper_drive_db=max(0.45, min(1.65, desired_lift * 0.34)),
                clipper_mix=0.30 if policy == "commercial" else 0.38,
                parallel_density_mix=0.075 if policy == "commercial" else 0.095,
                limiter_gain_db=desired_lift * 0.28,
                codec_safe=False,
            ),
        ),
        (
            "clipper_limiter_candidate",
            dict(
                target_tp=target_tp,
                target_lift_db=desired_lift,
                density_drive_amount=0.18,
                clipper_drive_db=max(0.65, min(2.10, desired_lift * 0.48)),
                clipper_mix=0.42,
                parallel_density_mix=0.045,
                limiter_gain_db=desired_lift * 0.34,
                codec_safe=False,
            ),
        ),
        (
            "codec_safe_loud_candidate",
            dict(
                target_tp=max(-0.85, min(-0.60, target_tp - 0.55)),
                target_lift_db=max(0.15, desired_lift * 0.62),
                density_drive_amount=0.12,
                clipper_drive_db=max(0.25, min(1.05, desired_lift * 0.24)),
                clipper_mix=0.22,
                parallel_density_mix=0.035,
                limiter_gain_db=desired_lift * 0.18,
                codec_safe=True,
            ),
        ),
        (
            "maximum_push_candidate",
            dict(
                target_tp=target_tp,
                target_lift_db=desired_lift * (1.12 if policy == "maximum" else 1.02),
                density_drive_amount=0.36 if policy == "maximum" else 0.30,
                clipper_drive_db=max(0.85, min(2.45, desired_lift * 0.58)),
                clipper_mix=0.48 if policy == "maximum" else 0.42,
                parallel_density_mix=0.115 if policy == "maximum" else 0.085,
                limiter_gain_db=desired_lift * (0.42 if policy == "maximum" else 0.36),
                codec_safe=False,
            ),
        ),
    ]
    for name, kwargs in specs:
        try:
            audio, meta = _build_candidate(name, x, sr, oversample=oversample, analyzer=analyzer, **kwargs)
            advisories, blockers = _damage_advisories(base_analysis, meta.get("analysis", {}), policy)
            meta["damage_advisories"] = advisories
            meta["blocking_reasons"] = blockers
            meta["warning_blocking"] = bool(blockers)
            meta["warning_advisory"] = bool(advisories and not blockers)
            meta["score"] = _candidate_score(meta, base_analysis, kwargs["target_tp"], policy) - len(blockers) * 999.0
            candidates.append((name, audio, meta))
        except Exception as exc:
            candidates.append((name, x, {"candidate_id": name, "actual_audio_rendered": False, "error": str(exc)[:300], "score": -9999.0}))

    # Select: Commercial prefers commercial density if close; Maximum prefers the
    # max push candidate.  Severe blocks can still force a cleaner fallback.
    metas = [m for _, _, m in candidates]
    deliverable = [(name, audio, meta) for name, audio, meta in candidates if not meta.get("blocking_reasons") and meta.get("actual_audio_rendered")]
    if not deliverable:
        report.update({
            "applied": False,
            "reason": "all_loud_candidates_blocked_by_severe_damage",
            "candidates": [{k: v for k, v in m.items() if k != "analysis"} for m in metas],
            "blocking_reasons": sorted({b for m in metas for b in (m.get("blocking_reasons") or [])}),
            "warning_blocking": True,
            "warning_advisory": False,
        })
        return x, report

    # Main clean is deliverable but should only win if all loud candidates are
    # severely blocked.  Exclude it from normal Commercial/Maximum selection.
    loud_deliverable = [(n, a, m) for n, a, m in deliverable if n != "main_clean_candidate"] or deliverable
    loud_deliverable.sort(key=lambda item: float(item[2].get("score", -9999.0) or -9999.0), reverse=True)
    selected_name, selected_audio, selected_meta = loud_deliverable[0]
    if policy == "maximum":
        max_items = [item for item in loud_deliverable if item[0] == "maximum_push_candidate"]
        if max_items and float(max_items[0][2].get("score", -9999.0) or -9999.0) > -100.0:
            selected_name, selected_audio, selected_meta = max_items[0]
    elif policy == "commercial":
        comm_items = [item for item in loud_deliverable if item[0] == "commercial_density_candidate"]
        if comm_items:
            comm_score = float(comm_items[0][2].get("score", -9999.0) or -9999.0)
            best_score = float(selected_meta.get("score", -9999.0) or -9999.0)
            if comm_score >= best_score - 4.0:
                selected_name, selected_audio, selected_meta = comm_items[0]

    selected_analysis = selected_meta.get("analysis") if isinstance(selected_meta.get("analysis"), dict) else _analysis_with_true_peak(selected_audio, sr, analyzer, oversample)
    selected_metrics = _summary(selected_analysis)
    for m in metas:
        m["selected_delivery_candidate"] = bool(m.get("candidate_id") == selected_name)
        m["accepted"] = bool(m.get("candidate_id") == selected_name)
        m["rejected"] = bool(m.get("candidate_id") != selected_name)
        m["quality_gate"] = "fail" if m.get("blocking_reasons") else "caution" if m.get("damage_advisories") else "pass"
        m["quality_rejected"] = bool(m.get("blocking_reasons"))

    all_advisories = []
    for m in metas:
        for adv in m.get("damage_advisories", []) or []:
            if isinstance(adv, dict):
                all_advisories.append({**adv, "candidate_id": m.get("candidate_id")})

    report.update({
        "applied": bool(selected_name != "main_clean_candidate"),
        "reason": "commercial_maximum_loud_candidate_selected" if selected_name != "main_clean_candidate" else "only_clean_candidate_deliverable",
        "selected_candidate_id": selected_name,
        "selected_reason": f"{policy}_mode_prioritizes_actual_loud_wav_candidate_warning_advisory_unless_severe",
        "selected_candidate_metrics": selected_metrics,
        "clean_candidate_metrics": base_metrics,
        "loud_candidate_metrics": selected_metrics,
        "final_true_peak_dbtp": selected_metrics.get("true_peak_dbtp"),
        "under_pushed": bool(float(selected_metrics.get("true_peak_dbtp", -99.0) or -99.0) <= -1.0),
        "push_gain_db": selected_meta.get("push_gain_db"),
        "clipper_drive_db": selected_meta.get("clipper_drive_db"),
        "limiter_gain_db": selected_meta.get("limiter_gain_db"),
        "density_drive_amount": selected_meta.get("density_drive_amount"),
        "parallel_density_mix": selected_meta.get("parallel_density_mix"),
        "damage_advisories": all_advisories,
        "blocking_reasons": sorted({b for m in metas for b in (m.get("blocking_reasons") or [])}),
        "warning_blocking": False,
        "warning_advisory": bool(all_advisories),
        "codec_safe_vs_maximum_wav_decision": {
            "maximum_wav_master_candidate": "selectable_for_commercial_or_maximum_delivery",
            "codec_safe_distribution_candidate": "rendered_and_reported_separately_not_allowed_to_block_wav_maximum_master_by_itself",
            "selected": selected_name,
        },
        "true_peak_target_honored": bool(-0.30 <= float(selected_metrics.get("true_peak_dbtp", -99.0) or -99.0) <= -0.05),
        "candidates": [{k: v for k, v in m.items() if k != "analysis"} for m in metas],
        "selected_analysis": selected_analysis,
    })
    return np.nan_to_num(selected_audio if report["applied"] else x, nan=0.0, posinf=0.0, neginf=0.0), report
