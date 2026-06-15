from __future__ import annotations

import os
from typing import Any

import numpy as np
from scipy.signal import resample_poly


def _env_bool(name: str, default: bool = True) -> bool:
    val = os.environ.get(name)
    if val is None:
        return bool(default)
    return str(val).strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    try:
        val = float(os.environ.get(name, str(default)))
    except Exception:
        val = float(default)
    if lo is not None:
        val = max(float(lo), val)
    if hi is not None:
        val = min(float(hi), val)
    return float(val)


def _db_to_lin(db: float) -> float:
    return float(10.0 ** (float(db) / 20.0))


def _lin_to_db(x: Any, eps: float = 1e-12) -> Any:
    return 20.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), eps))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return float(default)


def _z(v: float, center: float, scale: float) -> float:
    return float((float(v) - float(center)) / max(float(scale), 1e-9))


def _ensure_stereo(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.stack([arr, arr], axis=1)
    if arr.shape[1] == 1:
        arr = np.repeat(arr, 2, axis=1)
    return np.nan_to_num(arr[:, :2], nan=0.0, posinf=0.0, neginf=0.0)


def _mono_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    m = np.mean(x, axis=1) if x.ndim == 2 else x
    return float(np.sqrt(np.mean(np.square(m, dtype=np.float64)) + 1e-12))


def _crest_db(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    peak = float(np.max(np.abs(x)))
    rms = _mono_rms(x)
    return float(20.0 * np.log10(max(peak, 1e-12) / max(rms, 1e-12)))


def _rms_db(x: np.ndarray) -> float:
    return float(20.0 * np.log10(max(_mono_rms(x), 1e-12)))


def _midside(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    st = _ensure_stereo(x)
    mid = 0.5 * (st[:, 0] + st[:, 1])
    side = 0.5 * (st[:, 0] - st[:, 1])
    return mid, side


def _band_energy_fft(sig: np.ndarray, sr: int, lo: float, hi: float) -> float:
    if sig.size < 16:
        return 0.0
    n = int(2 ** np.ceil(np.log2(max(256, min(sig.size, 262144)))))
    if sig.size > n:
        # Center-crop long windows so FFT cost stays bounded.
        start = (sig.size - n) // 2
        sig = sig[start:start + n]
    win = np.hanning(sig.size).astype(np.float64)
    spec = np.fft.rfft(sig.astype(np.float64) * win, n=n)
    mag2 = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    mask = (freqs >= float(lo)) & (freqs < float(hi))
    if not np.any(mask):
        return 0.0
    return float(np.mean(mag2[mask]) + 1e-18)


def _spectral_flatness_fft(sig: np.ndarray, sr: int, lo: float, hi: float) -> float:
    if sig.size < 16:
        return 0.0
    n = int(2 ** np.ceil(np.log2(max(256, min(sig.size, 262144)))))
    if sig.size > n:
        start = (sig.size - n) // 2
        sig = sig[start:start + n]
    win = np.hanning(sig.size).astype(np.float64)
    spec = np.fft.rfft(sig.astype(np.float64) * win, n=n)
    mag = np.abs(spec) + 1e-12
    freqs = np.fft.rfftfreq(n, 1.0 / float(sr))
    mask = (freqs >= float(lo)) & (freqs < float(hi))
    vals = mag[mask]
    if vals.size == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))) / (np.mean(vals) + 1e-12))


def _band_ratio_db(num: float, den: float) -> float:
    return float(10.0 * np.log10((float(num) + 1e-18) / (float(den) + 1e-18)))


def _section_metrics(x: np.ndarray, sr: int) -> dict[str, float]:
    st = _ensure_stereo(x)
    mid, side = _midside(st)
    mono = np.mean(st, axis=1)
    rms = _mono_rms(st)
    peak = float(np.max(np.abs(st))) if st.size else 0.0
    crest = float(20.0 * np.log10(max(peak, 1e-12) / max(rms, 1e-12)))
    total = _band_energy_fft(mono, sr, 20, min(20000, sr / 2 - 100))
    low = _band_energy_fft(mono, sr, 20, 120)
    low_m = _band_energy_fft(mid, sr, 20, 120)
    low_s = _band_energy_fft(side, sr, 20, 120)
    harsh = _band_energy_fft(mono, sr, 2500, 5000)
    sibil = _band_energy_fft(mono, sr, 5000, 8000)
    air = _band_energy_fft(mono, sr, 8000, min(16000, sr / 2 - 100))
    side_air = _band_energy_fft(side, sr, 8000, min(16000, sr / 2 - 100))
    side_total = float(np.mean(np.square(side, dtype=np.float64)) + 1e-18)
    mid_total = float(np.mean(np.square(mid, dtype=np.float64)) + 1e-18)
    # Simple transient proxy: mean positive derivative over RMS.
    d = np.diff(mono.astype(np.float64), prepend=mono[:1].astype(np.float64))
    onset_density = float(np.mean(np.maximum(d, 0.0)) / max(rms, 1e-8))
    # Correlation is important for codec/mono risk.
    corr = 1.0
    try:
        corr = float(np.corrcoef(st[:, 0], st[:, 1])[0, 1])
        if not np.isfinite(corr):
            corr = 0.0
    except Exception:
        corr = 0.0
    return {
        "rms_db": _rms_db(st),
        "peak_dbfs": float(_lin_to_db(peak)),
        "crest_db": crest,
        "low_ratio": float(low / (total + 1e-18)),
        "low_db": float(10.0 * np.log10(low + 1e-18)),
        "low_side_ratio_db": _band_ratio_db(low_s, low_m),
        "side_ratio_db": _band_ratio_db(side_total, mid_total),
        "side_air_ratio_db": _band_ratio_db(side_air, air),
        "air_ratio": float(air / (total + 1e-18)),
        "harsh_ratio": float((harsh + sibil) / (total + 1e-18)),
        "high_flatness": _spectral_flatness_fft(mono, sr, 8000, min(16000, sr / 2 - 100)),
        "side_high_flatness": _spectral_flatness_fft(side, sr, 8000, min(16000, sr / 2 - 100)),
        "onset_density": onset_density,
        "correlation": corr,
    }


def _window_iter(y: np.ndarray, sr: int, win_sec: float, hop_sec: float):
    n = y.shape[0]
    win = max(1024, int(round(float(win_sec) * sr)))
    hop = max(512, int(round(float(hop_sec) * sr)))
    if n <= win:
        yield 0, n, y
        return
    for start in range(0, n - win + 1, hop):
        yield start, start + win, y[start:start + win]
    if n - win > 0:
        yield n - win, n, y[n - win:n]


def _top_windows(scored: list[dict[str, Any]], k: int = 2, min_spacing_sec: float = 2.0) -> list[dict[str, Any]]:
    scored = sorted(scored, key=lambda d: float(d.get("score", -1e9)), reverse=True)
    out: list[dict[str, Any]] = []
    for item in scored:
        mid = 0.5 * (float(item.get("start_sec", 0.0)) + float(item.get("end_sec", 0.0)))
        if all(abs(mid - 0.5 * (float(o.get("start_sec", 0.0)) + float(o.get("end_sec", 0.0)))) >= min_spacing_sec for o in out):
            out.append(item)
        if len(out) >= k:
            break
    return out


def _representative_section_map(y: np.ndarray, sr: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    y = _ensure_stereo(y)
    duration = float(y.shape[0] / max(sr, 1))
    classes = {
        "loudest": {"win": 3.0, "hop": 0.75, "k": 2},
        "true_peak_risk": {"win": 2.0, "hop": 0.50, "k": 2},
        "dense_chorus_drop": {"win": 4.0, "hop": 1.00, "k": 2},
        "low_end_stress": {"win": 3.0, "hop": 0.75, "k": 2},
        "side_fizz": {"win": 3.0, "hop": 0.75, "k": 2},
        "vocal_harshness_proxy": {"win": 2.0, "hop": 0.50, "k": 1},
        "transient_stress": {"win": 2.0, "hop": 0.50, "k": 1},
        "codec_risk": {"win": 3.0, "hop": 0.75, "k": 1},
        "final_climax": {"win": 6.0, "hop": 1.00, "k": 1},
    }
    selected: list[dict[str, Any]] = []
    debug_scores: dict[str, Any] = {}
    cache: dict[tuple[float, float, int, int], dict[str, float]] = {}

    def metrics_for(start: int, end: int, seg: np.ndarray) -> dict[str, float]:
        key = (round(start / sr, 3), round(end / sr, 3), start, end)
        if key not in cache:
            cache[key] = _section_metrics(seg, sr)
        return cache[key]

    for cls, cfg in classes.items():
        scored: list[dict[str, Any]] = []
        for start, end, seg in _window_iter(y, sr, cfg["win"], cfg["hop"]):
            m = metrics_for(start, end, seg)
            pos = float((0.5 * (start + end)) / max(y.shape[0], 1))
            if cls == "loudest":
                score = 0.58 * m["rms_db"] - 0.12 * m["crest_db"] + 8.0 * m["low_ratio"]
            elif cls == "true_peak_risk":
                score = 0.52 * m["peak_dbfs"] + 0.20 * m["crest_db"] + 0.18 * m["air_ratio"] * 100.0 + 0.10 * max(0.0, m["side_ratio_db"])
            elif cls == "dense_chorus_drop":
                score = 0.40 * m["rms_db"] - 0.32 * m["crest_db"] + 0.20 * m["onset_density"] + 0.08 * (m["low_ratio"] + m["air_ratio"]) * 50.0
            elif cls == "low_end_stress":
                score = 1.15 * m["low_ratio"] * 100.0 + 0.32 * m["low_side_ratio_db"] - 0.10 * m["crest_db"]
            elif cls == "side_fizz":
                score = 0.65 * m["side_air_ratio_db"] + 0.55 * max(0.0, m["side_ratio_db"]) + 18.0 * m["side_high_flatness"] + 10.0 * m["air_ratio"]
            elif cls == "vocal_harshness_proxy":
                score = 80.0 * m["harsh_ratio"] + 10.0 * m["air_ratio"] + 0.10 * m["rms_db"]
            elif cls == "transient_stress":
                score = 0.42 * m["onset_density"] + 0.35 * m["crest_db"] + 0.23 * m["peak_dbfs"]
            elif cls == "codec_risk":
                score = 0.40 * m["peak_dbfs"] + 0.25 * m["side_air_ratio_db"] + 15.0 * m["high_flatness"] + 0.15 * (1.0 - m["correlation"]) * 10.0
            else:  # final_climax
                score = 0.35 * m["rms_db"] + 0.25 * pos * 20.0 + 0.20 * m["low_ratio"] * 100.0 + 0.20 * max(0.0, m["side_ratio_db"])
                if pos < 0.60:
                    score -= 20.0
            scored.append({
                "class": cls,
                "start_sec": round(start / sr, 3),
                "end_sec": round(end / sr, 3),
                "score": round(float(score), 4),
                "metrics": {k: round(float(v), 5) for k, v in m.items()},
            })
        top = _top_windows(scored, k=int(cfg["k"]), min_spacing_sec=1.5)
        selected.extend(top)
        debug_scores[cls] = [{k: v for k, v in t.items() if k != "metrics"} for t in top]

    # Deduplicate overlaps while preserving multi-tags.
    dedup: list[dict[str, Any]] = []
    for item in sorted(selected, key=lambda d: (float(d.get("start_sec", 0.0)), -float(d.get("score", 0.0)))):
        merged = False
        for out in dedup:
            overlap = max(0.0, min(float(out["end_sec"]), float(item["end_sec"])) - max(float(out["start_sec"]), float(item["start_sec"])))
            shorter = max(1e-6, min(float(out["end_sec"]) - float(out["start_sec"]), float(item["end_sec"]) - float(item["start_sec"])))
            if overlap / shorter >= 0.50:
                out.setdefault("tags", [])
                if item["class"] not in out["tags"]:
                    out["tags"].append(item["class"])
                out["score"] = round(max(float(out.get("score", 0.0)), float(item.get("score", 0.0))), 4)
                # Keep strongest metric map for debugging.
                if float(item.get("score", 0.0)) >= float(out.get("score", 0.0)):
                    out["metrics"] = item.get("metrics", out.get("metrics", {}))
                merged = True
                break
        if not merged:
            item = dict(item)
            item["tags"] = [item.pop("class")]
            dedup.append(item)
    max_sections = int(_env_float("BUSY_VIRTUAL_HOT_MAX_SECTIONS", 12, 4, 20))
    dedup = sorted(dedup, key=lambda d: float(d.get("score", 0.0)), reverse=True)[:max_sections]
    dedup = sorted(dedup, key=lambda d: float(d.get("start_sec", 0.0)))
    return dedup, {
        "schema_version": "busy_representative_section_map_v8_5_3_17",
        "duration_sec": round(duration, 3),
        "section_count": len(dedup),
        "classes": list(classes.keys()),
        "selection_policy": "top-k risk windows per class, overlap-deduplicated with multi-tags",
        "top_by_class": debug_scores,
        "sections": dedup,
    }


def _soft_clip_tanh(x: np.ndarray, drive: float = 1.25) -> np.ndarray:
    drive = float(np.clip(drive, 0.5, 3.0))
    return np.tanh(drive * x) / max(np.tanh(drive), 1e-12)


def _virtual_section_at_gain(seg: np.ndarray, sr: int, gain_db: float, ceiling_db: float, os_factor: int, clip_mix: float, clip_drive: float) -> dict[str, Any]:
    seg = _ensure_stereo(seg)
    xg = seg.astype(np.float32) * _db_to_lin(gain_db)
    try:
        xos = resample_poly(xg, up=int(os_factor), down=1, axis=0)
        os_sr = int(sr * os_factor)
    except Exception:
        xos = xg
        os_sr = sr
    xclip = _soft_clip_tanh(xos, drive=clip_drive)
    xpre = (1.0 - clip_mix) * xos + clip_mix * xclip
    peak_in = float(np.max(np.abs(xos))) if xos.size else 0.0
    ceiling_lin = _db_to_lin(ceiling_db)
    scalar_gr = min(1.0, ceiling_lin / max(peak_in, 1e-12))
    # Static proxy limiting: enough for capacity/blocker estimation without full final render.
    yout = xpre * scalar_gr
    peak_out = float(np.max(np.abs(yout))) if yout.size else 0.0
    rms_in = _mono_rms(seg)
    rms_out = _mono_rms(yout)
    crest_out = _crest_db(yout)
    # LUFS proxy: RMS gain minus nonlinearity/GR penalty; final QC will verify.
    input_rms_db = 20.0 * np.log10(max(rms_in, 1e-12))
    out_rms_db = 20.0 * np.log10(max(rms_out, 1e-12))
    lufs_gain_est = float(out_rms_db - input_rms_db)
    gr_db = float(-20.0 * np.log10(max(scalar_gr, 1e-12)))
    clip_ratio = float(np.mean(np.abs(xos) > ceiling_lin)) if xos.size else 0.0
    metrics = _section_metrics(yout.astype(np.float32), os_sr if os_sr <= 96000 else sr * os_factor)
    return {
        "input_gain_db": round(float(gain_db), 3),
        "peak_in_dbtp_est": round(float(_lin_to_db(peak_in)), 3),
        "peak_out_dbtp_est": round(float(_lin_to_db(peak_out)), 3),
        "scalar_gain_reduction_db_est": round(gr_db, 3),
        "clip_ratio_est": round(clip_ratio, 6),
        "crest_out_db_est": round(float(crest_out), 3),
        "lufs_gain_est_db": round(lufs_gain_est, 3),
        "section_metrics_after_proxy": {k: round(float(v), 5) for k, v in metrics.items()},
    }



def _nested_dict(obj: Any, *keys: str) -> dict[str, Any]:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict):
            return {}
        cur = cur.get(key)
    return cur if isinstance(cur, dict) else {}


def _extract_role_identity_context(before: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    """Build a compact role/protection map for the virtual premaster planner.

    The virtual solver should not only solve risks.  It should preserve the
    musical identity that the normal Stage A decision already discovered: vocal
    spine, kick/bass drive, intentional side texture, genre/profile role, and
    Element Action Manual posture.  This map is deliberately compact so it can be
    copied to stage_state without exposing private rulepack contents.
    """
    before = before if isinstance(before, dict) else {}
    decision = decision if isinstance(decision, dict) else {}
    contract = (
        _nested_dict(before, "role_first_dsp_contract")
        or _nested_dict(before, "input_analysis", "role_first_dsp_contract")
        or _nested_dict(decision, "role_first_dsp_contract")
        or _nested_dict(decision, "final_decision_context", "role_first_dsp_contract")
    )
    role_map = _nested_dict(contract, "role_map") or _nested_dict(before, "role_map") or _nested_dict(decision, "role_map")
    role_importance = _nested_dict(contract, "role_importance") or _nested_dict(role_map, "role_importance")
    role_scores = _nested_dict(role_map, "role_scores") or role_importance
    role_confidence = _nested_dict(role_map, "role_confidence")

    def r(name: str, default: float = 0.0) -> float:
        # Importance is preferred because it already combines score/function/confidence.
        return float(np.clip(_safe_float(role_importance.get(name, role_scores.get(name, default))), 0.0, 1.0))

    vocal_focus = max(r("vocal_front_focus"), r("vocal_chant_or_hook"), 0.55 * r("vocal_embedded_texture"))
    low_drive = max(r("club_grid_energy"), r("four_on_floor_pulse"), r("riff_low_end_articulation"), r("low_string_or_chug_energy"))
    low_stability_need = max(r("low_end_tightness_need"), r("sub_mono_stability_need"), r("kick_bass_separation_need"))
    side_texture_identity = max(r("ambient_side_texture_preservation"), r("industrial_noise_or_grit"), r("glitch_or_cutup_texture"), r("psychedelic_or_warped_space"), r("synth_edge_or_brightness"))
    top_risk = max(r("brittle_top_or_ai_shimmer_risk"), r("upper_mid_harshness_risk"), r("vocal_sibilance_risk"))
    transient_need = max(r("transient_preservation_need"), r("live_drum_or_band_punch"), r("four_on_floor_pulse"))
    codec_need = r("codec_translation_need")
    gloss_need = r("commercial_gloss_need")

    profile_blend = decision.get("profile_blend") if isinstance(decision.get("profile_blend"), list) else []
    genre_mix = decision.get("genre_mix") if isinstance(decision.get("genre_mix"), list) else []
    primary_profiles = []
    for item in (profile_blend or genre_mix):
        if isinstance(item, dict):
            primary_profiles.append({
                "profile": item.get("profile"),
                "role": item.get("role") or item.get("role_focus"),
                "weight": item.get("weight"),
            })
        if len(primary_profiles) >= 4:
            break

    chain = decision.get("mastering_chain") if isinstance(decision.get("mastering_chain"), dict) else {}
    ecounts = _nested_dict(contract, "element_action_counts") or _nested_dict(contract, "element_action_manual", "action_counts")
    preserve = contract.get("preserve", []) if isinstance(contract.get("preserve"), list) else []
    control = contract.get("control", []) if isinstance(contract.get("control"), list) else []

    protection_constraints: list[str] = []
    if vocal_focus >= 0.45:
        protection_constraints.append("protect_vocal_presence_and_air")
    if low_drive >= 0.45:
        protection_constraints.append("preserve_kick_bass_drive")
    if side_texture_identity >= 0.40:
        protection_constraints.append("preserve_intentional_side_texture")
    if transient_need >= 0.35:
        protection_constraints.append("protect_transient_punch")
    if gloss_need >= 0.35:
        protection_constraints.append("avoid_over_dulling_commercial_gloss")

    return {
        "schema_version": "busy_musical_role_identity_map_v8_5_3_17",
        "enabled": bool(contract or role_importance or role_scores),
        "mastering_intent": decision.get("mastering_intent"),
        "selected_profile": decision.get("selected_profile"),
        "selected_family": decision.get("selected_family"),
        "primary_profiles": primary_profiles,
        "role_strengths": {
            "vocal_focus": round(vocal_focus, 4),
            "low_end_drive": round(low_drive, 4),
            "low_end_stability_need": round(low_stability_need, 4),
            "side_texture_identity": round(side_texture_identity, 4),
            "top_risk_or_fatigue": round(top_risk, 4),
            "transient_preservation_need": round(transient_need, 4),
            "codec_translation_need": round(codec_need, 4),
            "commercial_gloss_need": round(gloss_need, 4),
        },
        "protection_constraints": protection_constraints,
        "element_action_counts": ecounts,
        "preserve_count": len(preserve),
        "control_count": len(control),
        "decision_chain_summary": {
            "corrective_eq_count": len(chain.get("corrective_eq", []) or []) if isinstance(chain.get("corrective_eq", []), list) else 0,
            "character_eq_count": len(chain.get("character_eq", []) or []) if isinstance(chain.get("character_eq", []), list) else 0,
            "dynamic_eq_count": len(chain.get("dynamic_eq", []) or []) if isinstance(chain.get("dynamic_eq", []), list) else 0,
            "ms_strategy": chain.get("ms_strategy", {}) if isinstance(chain.get("ms_strategy", {}), dict) else {},
            "limiter": chain.get("limiter", {}) if isinstance(chain.get("limiter", {}), dict) else {},
            "loudness_target": chain.get("loudness_target", {}) if isinstance(chain.get("loudness_target", {}), dict) else {},
        },
        "policy": "Virtual solver uses base analysis and role/profile/protection context to test premaster candidates; it must not solve blockers by damaging protected musical identity.",
    }


def _role_strength(role_ctx: dict[str, Any], key: str, default: float = 0.0) -> float:
    strengths = role_ctx.get("role_strengths", {}) if isinstance(role_ctx.get("role_strengths", {}), dict) else {}
    return float(np.clip(_safe_float(strengths.get(key), default), 0.0, 1.0))

def _blockers_for_virtual_result(
    v: dict[str, Any],
    section: dict[str, Any],
    base_metrics: dict[str, float],
    profile: dict[str, Any],
    before: dict[str, Any],
    role_ctx: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    role_ctx = role_ctx if isinstance(role_ctx, dict) else {}
    vocal_protect = _role_strength(role_ctx, "vocal_focus")
    low_drive = _role_strength(role_ctx, "low_end_drive")
    low_stability_need = _role_strength(role_ctx, "low_end_stability_need")
    side_texture_identity = _role_strength(role_ctx, "side_texture_identity")
    top_risk_or_fatigue = _role_strength(role_ctx, "top_risk_or_fatigue")
    transient_protect = _role_strength(role_ctx, "transient_preservation_need")
    codec_need = _role_strength(role_ctx, "codec_translation_need")
    gain = _safe_float(v.get("input_gain_db"))
    crest = _safe_float(v.get("crest_out_db_est"), 99.0)
    gr = _safe_float(v.get("scalar_gain_reduction_db_est"), 0.0)
    clip_ratio = _safe_float(v.get("clip_ratio_est"), 0.0)
    m = v.get("section_metrics_after_proxy", {}) if isinstance(v.get("section_metrics_after_proxy"), dict) else {}
    crest_min = _safe_float(profile.get("hard_crest_min_db"), 4.9)
    soft_crest = _safe_float(profile.get("crest_min_db"), 5.7)
    if gr >= 3.0:
        blockers.append({"type": "true_peak_pressure", "severity": min(1.0, gr / 5.5), "solvable": True, "recipe": "clipper_limiter_distribution_adjust", "evidence": {"gain_reduction_db_est": round(gr, 3)}})
    if clip_ratio >= 0.020:
        blockers.append({"type": "clip_density_overload", "severity": min(1.0, clip_ratio / 0.08), "solvable": True, "recipe": "soft_clip_distribution_adjust", "evidence": {"clip_ratio_est": round(clip_ratio, 5)}})
    if crest < crest_min:
        blockers.append({"type": "crest_collapse", "severity": min(1.0, (crest_min - crest) / 2.2), "solvable": False if crest < crest_min - 1.0 else True, "recipe": "transient_protection_or_stop", "evidence": {"crest_out_db_est": round(crest, 3), "hard_crest_min_db": round(crest_min, 3)}})
    elif crest < soft_crest:
        blockers.append({"type": "crest_warning", "severity": min(0.8, (soft_crest - crest) / 2.0), "solvable": True, "recipe": "transient_protection", "evidence": {"crest_out_db_est": round(crest, 3), "soft_crest_min_db": round(soft_crest, 3)}})
    low_ratio = _safe_float(m.get("low_ratio"), _safe_float(base_metrics.get("low_ratio")))
    low_side = _safe_float(m.get("low_side_ratio_db"), _safe_float(base_metrics.get("low_side_ratio_db")))
    if low_ratio > 0.37 or low_side > -13.0:
        low_sev = min(1.0, max(_z(low_ratio, 0.34, 0.20), _z(low_side, -14.0, 10.0)) * (0.90 + 0.25 * low_stability_need))
        blockers.append({"type": "low_end_bloom" if low_ratio > 0.37 else "low_side_instability", "severity": low_sev, "solvable": True, "recipe": "low_end_dynamic_eq_and_low_side_cleanup", "protection": "preserve_kick_bass_drive" if low_drive >= 0.45 else None, "evidence": {"low_ratio": round(low_ratio, 5), "low_side_ratio_db": round(low_side, 3), "low_end_drive_role": round(low_drive, 4), "low_stability_need": round(low_stability_need, 4)}})
    side_air = _safe_float(m.get("side_air_ratio_db"), _safe_float(base_metrics.get("side_air_ratio_db")))
    side_flat = _safe_float(m.get("side_high_flatness"), _safe_float(base_metrics.get("side_high_flatness")))
    air_ratio = _safe_float(m.get("air_ratio"), _safe_float(base_metrics.get("air_ratio")))
    if side_air > -4.0 or side_flat > 0.34 or air_ratio > 0.13:
        side_sev_raw = max(_z(side_air, -6.0, 9.0), _z(side_flat, 0.28, 0.35), _z(air_ratio, 0.10, 0.12))
        # Intentional side texture should be protected, but brittle-top risk keeps the control eligible.
        side_sev = min(1.0, side_sev_raw * (1.0 + 0.22 * top_risk_or_fatigue - 0.18 * side_texture_identity))
        blockers.append({"type": "side_high_fizz", "severity": side_sev, "solvable": True, "recipe": "side_high_dynamic_shelf", "protection": "preserve_intentional_side_texture" if side_texture_identity >= 0.40 else None, "evidence": {"side_air_ratio_db": round(side_air, 3), "side_high_flatness": round(side_flat, 5), "air_ratio": round(air_ratio, 5), "side_texture_identity": round(side_texture_identity, 4), "top_risk_or_fatigue": round(top_risk_or_fatigue, 4)}})
    harsh_ratio = _safe_float(m.get("harsh_ratio"), _safe_float(base_metrics.get("harsh_ratio")))
    if harsh_ratio > 0.18 and gain >= 1.0:
        harsh_sev = min(1.0, _z(harsh_ratio, 0.15, 0.18) * (1.0 + 0.18 * top_risk_or_fatigue))
        blockers.append({"type": "vocal_or_upper_mid_harshness", "severity": harsh_sev, "solvable": True, "recipe": "vocal_harshness_dynamic_eq", "protection": "protect_vocal_presence_and_air" if vocal_protect >= 0.45 else None, "evidence": {"harsh_ratio": round(harsh_ratio, 5), "vocal_focus_role": round(vocal_protect, 4), "top_risk_or_fatigue": round(top_risk_or_fatigue, 4)}})
    corr = _safe_float(m.get("correlation"), _safe_float(base_metrics.get("correlation"), 1.0))
    if corr < 0.18 and gain >= 1.0:
        blockers.append({"type": "stereo_correlation_risk", "severity": min(1.0, (0.18 - corr) / 0.35), "solvable": True, "recipe": "ms_width_stabilization", "evidence": {"correlation": round(corr, 4)}})
    if gain >= 2.0 and (_safe_float(v.get("peak_out_dbtp_est"), -99) > -0.35 or side_air > -5.0 or air_ratio > 0.12):
        blockers.append({"type": "codec_stress", "severity": min(1.0, 0.35 + gain / 8.0 + 0.12 * codec_need), "solvable": True, "recipe": "codec_protection_ceiling_and_hf_control", "evidence": {"gain_db": round(gain, 3), "peak_out_dbtp_est": v.get("peak_out_dbtp_est"), "side_air_ratio_db": round(side_air, 3), "codec_translation_need": round(codec_need, 4)}})
    # Merge duplicate types, keep strongest.
    merged: dict[str, dict[str, Any]] = {}
    for b in blockers:
        t = str(b.get("type"))
        if t not in merged or _safe_float(b.get("severity")) > _safe_float(merged[t].get("severity")):
            merged[t] = b
    return sorted(merged.values(), key=lambda d: _safe_float(d.get("severity")), reverse=True)


def _solution_plan_from_blockers(blockers: list[dict[str, Any]], gain_db: float, profile: dict[str, Any], role_ctx: dict[str, Any] | None = None) -> dict[str, Any]:
    role_ctx = role_ctx if isinstance(role_ctx, dict) else {}
    vocal_protect = _role_strength(role_ctx, "vocal_focus")
    low_drive = _role_strength(role_ctx, "low_end_drive")
    low_stability_need = _role_strength(role_ctx, "low_end_stability_need")
    side_texture_identity = _role_strength(role_ctx, "side_texture_identity")
    transient_protect = _role_strength(role_ctx, "transient_preservation_need")
    gloss_need = _role_strength(role_ctx, "commercial_gloss_need")
    static_moves: list[dict[str, Any]] = []
    dynamic_moves: list[dict[str, Any]] = []
    notes: list[str] = []
    low_side_mono_amount = 0.0
    clip_drive_delta = 0.0
    clip_mix_delta = 0.0
    ceiling_delta_db = 0.0
    stop_reasons: list[str] = []
    for b in blockers:
        t = str(b.get("type"))
        sev = float(np.clip(_safe_float(b.get("severity"), 0.0), 0.0, 1.0))
        if not b.get("solvable", True):
            stop_reasons.append(t)
            continue
        if t in {"low_end_bloom", "low_side_instability"}:
            low_cut_scalar = float(np.clip(1.0 - 0.28 * low_drive, 0.62, 1.0))
            low_mono_bonus = float(np.clip(0.10 * low_stability_need, 0.0, 0.12))
            low_side_mono_amount = max(low_side_mono_amount, float(np.clip(0.38 + sev * 0.24 + low_mono_bonus, 0.38, 0.84)))
            dynamic_moves.append({"type": "bell", "freq": 88, "gain_db": round(-(0.24 + sev * 0.46) * low_cut_scalar, 3), "q": 0.82, "target": "side", "source": "virtual_hot_solver_low_side", "role_policy": "preserve_kick_bass_drive" if low_drive >= 0.45 else "standard"})
            static_moves.append({"type": "bell", "freq": 135, "gain_db": round(-(0.08 + sev * 0.18) * low_cut_scalar, 3), "q": 0.90, "target": "side", "source": "virtual_hot_solver_low_bloom", "role_policy": "preserve_kick_bass_drive" if low_drive >= 0.45 else "standard"})
            notes.append("solve_low_end_or_low_side_before_push_role_aware")
        elif t == "side_high_fizz":
            side_scalar = float(np.clip(1.0 - 0.42 * side_texture_identity + 0.16 * (1.0 if gloss_need < 0.25 else 0.0), 0.46, 1.0))
            dynamic_moves.append({"type": "bell", "freq": 9300, "gain_db": round(-(0.18 + sev * 0.58) * side_scalar, 3), "q": 1.28, "target": "side", "source": "virtual_hot_solver_side_fizz", "role_policy": "preserve_intentional_side_texture" if side_texture_identity >= 0.40 else "standard"})
            dynamic_moves.append({"type": "bell", "freq": 14800, "gain_db": round(-(0.10 + sev * 0.32) * side_scalar, 3), "q": 0.90, "target": "side", "source": "virtual_hot_solver_air_hash", "role_policy": "preserve_commercial_gloss" if gloss_need >= 0.35 else "standard"})
            notes.append("solve_side_high_fizz_before_push_role_aware")
        elif t == "vocal_or_upper_mid_harshness":
            vocal_scalar = float(np.clip(1.0 - 0.34 * vocal_protect, 0.58, 1.0))
            dynamic_moves.append({"type": "bell", "freq": 4200, "gain_db": round(-(0.14 + sev * 0.36) * vocal_scalar, 3), "q": 1.45, "target": "mid", "source": "virtual_hot_solver_vocal_harshness", "role_policy": "protect_vocal_presence_and_air" if vocal_protect >= 0.45 else "standard"})
            notes.append("solve_vocal_or_upper_mid_harshness_before_push_role_aware")
        elif t == "stereo_correlation_risk":
            static_moves.append({"type": "high_shelf", "freq": 7800, "gain_db": round(-(0.08 + sev * 0.22), 3), "q": 0.70, "target": "side", "source": "virtual_hot_solver_correlation"})
            notes.append("solve_stereo_correlation_before_push")
        elif t in {"true_peak_pressure", "clip_density_overload"}:
            clip_scalar = float(np.clip(1.0 - 0.30 * transient_protect, 0.62, 1.0))
            clip_drive_delta += float(np.clip(0.10 + sev * 0.30, 0.10, 0.40)) * clip_scalar
            clip_mix_delta += float(np.clip(0.010 + sev * 0.030, 0.010, 0.045)) * clip_scalar
            notes.append("redistribute_peak_pressure_to_clip_stage_role_aware")
        elif t == "crest_warning":
            notes.append("protect_transients_before_push")
        elif t == "codec_stress":
            ceiling_delta_db = min(ceiling_delta_db, -0.10 - sev * 0.25)
            dynamic_moves.append({"type": "bell", "freq": 11800, "gain_db": round(-(0.10 + sev * 0.26), 3), "q": 0.95, "target": "side", "source": "virtual_hot_solver_codec_hf"})
            notes.append("codec_protection_before_push")
    return {
        "static_moves": static_moves,
        "dynamic_moves": dynamic_moves,
        "low_side_mono_amount": round(float(low_side_mono_amount), 4),
        "clip_drive_delta_db": round(float(np.clip(clip_drive_delta, 0.0, 0.75)), 3),
        "clip_mix_delta": round(float(np.clip(clip_mix_delta, 0.0, 0.08)), 4),
        "ceiling_delta_db": round(float(np.clip(ceiling_delta_db, -0.55, 0.0)), 3),
        "notes": sorted(set(notes)),
        "role_aware": True,
        "role_adjustments": {
            "vocal_protect_scalar": round(vocal_protect, 4),
            "low_end_drive_scalar": round(low_drive, 4),
            "low_end_stability_need": round(low_stability_need, 4),
            "side_texture_identity_scalar": round(side_texture_identity, 4),
            "transient_protect_scalar": round(transient_protect, 4),
            "commercial_gloss_need": round(gloss_need, 4),
        },
        "protection_constraints": role_ctx.get("protection_constraints", []) if isinstance(role_ctx.get("protection_constraints", []), list) else [],
        "stop_reasons": sorted(set(stop_reasons)),
    }


def build_virtual_hot_commercial_solver(
    y: np.ndarray,
    sr: int,
    before: dict[str, Any] | None,
    decision: dict[str, Any] | None,
    mode: str,
    base_analysis: dict[str, Any] | None,
    profile: dict[str, Any] | None,
) -> dict[str, Any]:
    """Predict hot-commercial loudness capacity from representative sections.

    This is not the final limiter. It is a cheap capacity estimator and blocker
    solver: select high-risk sections, sweep virtual input gain through a proxy
    clip/limit model, predict first blockers, choose upstream recipes, and return
    a final push strategy for the real render path.
    """
    enabled = _env_bool("BUSY_VIRTUAL_HOT_SOLVER", True)
    report: dict[str, Any] = {
        "schema_version": "busy_virtual_hot_commercial_solver_v8_5_3_20_candidate_limiter_trace_fix",
        "active": False,
        "enabled": enabled,
        "mode": str(mode or ""),
        "policy": "role-aware virtual premaster and limiter strategy solver; tests correction candidates from base analysis before the real premaster/render; no user-defined max-push stop rule",
    }
    if not enabled:
        report["reason"] = "disabled"
        return report
    profile = profile if isinstance(profile, dict) else {}
    before = before if isinstance(before, dict) else {}
    base_analysis = base_analysis if isinstance(base_analysis, dict) else {}
    if not bool(profile.get("adaptive_hot_commercial") or profile.get("experimental_hot_commercial") or "commercial" in str(mode or "").lower() or "hot" in str(mode or "").lower()):
        report["reason"] = "non_hot_commercial_mode"
        return report
    try:
        y = _ensure_stereo(y)
        role_ctx = _extract_role_identity_context(before, decision if isinstance(decision, dict) else {})
        sections, section_report = _representative_section_map(y, sr)
        target_lufs = _safe_float(profile.get("target_lufs"), -8.8)
        floor_lufs = _safe_float(profile.get("commercial_floor_lufs"), target_lufs - 0.7)
        base_lufs = _safe_float(base_analysis.get("integrated_lufs"), -99.0)
        needed_to_floor = max(0.0, floor_lufs - base_lufs)
        needed_to_target = max(needed_to_floor, target_lufs - base_lufs)
        # v8.5.3.19: do not expose a user-chosen max-push as the definition of
        # hot-commercial capacity.  The solver sweeps until blockers remain
        # unsolved; the number below is only a non-user-facing crash guard in case
        # a proxy model fails to generate blockers.
        step = 0.25
        computational_guard_db = float(np.clip(max(needed_to_target + 4.0, 12.0), 12.0, 20.0))
        max_push = computational_guard_db
        os_factor = int(_env_float("BUSY_VIRTUAL_HOT_OS_FACTOR", 4, 2, 8))
        ceiling_db = _safe_float(profile.get("true_peak_ceiling_db"), _safe_float(profile.get("ceiling_db"), -1.0))
        # Do not assume -0.1 is codec-safe in the virtual planner; actual final ceiling remains engine-owned.
        planner_ceiling = min(ceiling_db, _env_float("BUSY_VIRTUAL_HOT_PLANNER_CEILING_DB", -0.65, -2.0, -0.1))
        clip_mix = _env_float("BUSY_VIRTUAL_HOT_PROXY_CLIP_MIX", 0.18, 0.0, 0.45)
        clip_drive = _env_float("BUSY_VIRTUAL_HOT_PROXY_CLIP_DRIVE", 1.25, 0.5, 3.0)
        gains = [round(float(g), 3) for g in np.arange(0.0, max_push + step * 0.5, step)]
        sweep_rows: list[dict[str, Any]] = []
        capacity_by_gain: dict[str, Any] = {}
        first_blocker: dict[str, Any] | None = None
        highest_clean_gain = 0.0
        highest_solvable_gain = 0.0
        best_gain = 0.0
        all_solution_blockers: list[dict[str, Any]] = []
        # Precompute actual section arrays only here; report stores time ranges only.
        for g in gains:
            gain_blockers: list[dict[str, Any]] = []
            worst: dict[str, Any] | None = None
            for idx, sec in enumerate(sections):
                start = int(round(float(sec.get("start_sec", 0.0)) * sr))
                end = int(round(float(sec.get("end_sec", 0.0)) * sr))
                if end <= start:
                    continue
                seg = y[max(0, start):min(y.shape[0], end)]
                base_m = sec.get("metrics", {}) if isinstance(sec.get("metrics"), dict) else _section_metrics(seg, sr)
                v = _virtual_section_at_gain(seg, sr, float(g), planner_ceiling, os_factor, clip_mix, clip_drive)
                blockers = _blockers_for_virtual_result(v, sec, base_m, profile, before, role_ctx=role_ctx)
                if blockers:
                    for b in blockers:
                        bb = dict(b)
                        bb.update({"section_index": idx, "section_tags": sec.get("tags", []), "section_start_sec": sec.get("start_sec"), "section_end_sec": sec.get("end_sec"), "gain_db": round(float(g), 3)})
                        gain_blockers.append(bb)
                    top = blockers[0]
                    if worst is None or _safe_float(top.get("severity")) > _safe_float(worst.get("severity")):
                        worst = dict(top)
                        worst.update({"section_index": idx, "section_tags": sec.get("tags", []), "section_start_sec": sec.get("start_sec"), "section_end_sec": sec.get("end_sec")})
                if len(sweep_rows) < 80:  # keep JSON bounded
                    sweep_rows.append({
                        "gain_db": round(float(g), 3),
                        "section_index": idx,
                        "section_tags": sec.get("tags", []),
                        "proxy": v,
                        "blockers": blockers[:3],
                    })
            hard = [b for b in gain_blockers if not b.get("solvable", True)]
            solvable = [b for b in gain_blockers if b.get("solvable", True)]
            if not gain_blockers:
                highest_clean_gain = float(g)
                highest_solvable_gain = float(g)
                best_gain = float(g)
            elif not hard:
                highest_solvable_gain = float(g)
                # Allow solvable blockers while keeping severity moderate.
                max_sev = max([_safe_float(b.get("severity")) for b in solvable] or [0.0])
                if max_sev <= 0.82:
                    best_gain = float(g)
                if first_blocker is None:
                    first_blocker = dict(worst or solvable[0])
                    first_blocker["gain_db"] = round(float(g), 3)
                all_solution_blockers.extend(solvable)
            else:
                if first_blocker is None:
                    first_blocker = dict(worst or hard[0])
                    first_blocker["gain_db"] = round(float(g), 3)
                all_solution_blockers.extend(gain_blockers)
                # Do not break; later rows are useful for capacity map/debugging.
            if abs(float(g) / max(step, 0.01) - round(float(g) / max(step, 0.01))) < 1e-6:
                capacity_by_gain[f"{float(g):.2f}"] = {
                    "blocker_count": len(gain_blockers),
                    "hard_blocker_count": len(hard),
                    "top_blocker": worst,
                }
        # Make sure the plan tries to close floor/target gap when proxy says possible.
        recommended_gain = max(best_gain, min(highest_solvable_gain, needed_to_target))
        if highest_solvable_gain <= 0.0:
            recommended_gain = min(max_push, max(0.0, min(needed_to_target, 0.75)))
        recommended_gain = float(np.clip(recommended_gain, 0.0, max_push))
        # Collect blockers up to recommended gain plus a small lookahead margin.
        plan_blockers = [b for b in all_solution_blockers if _safe_float(b.get("gain_db"), 0.0) <= recommended_gain + 0.50]
        plan_blockers = sorted(plan_blockers, key=lambda b: (_safe_float(b.get("severity"), 0.0), _safe_float(b.get("gain_db"), 0.0)), reverse=True)[:8]
        solution = _solution_plan_from_blockers(plan_blockers, recommended_gain, profile, role_ctx=role_ctx)
        if solution.get("stop_reasons") and recommended_gain > 0.25:
            recommended_gain = max(0.0, min(recommended_gain, _safe_float(first_blocker.get("gain_db") if first_blocker else recommended_gain, recommended_gain) - 0.25))
        expected_lufs = base_lufs + recommended_gain * 0.72 if base_lufs > -90 else None
        recommended_ceiling = float(np.clip(planner_ceiling + _safe_float(solution.get("ceiling_delta_db"), 0.0), -2.0, -0.10))
        report.update({
            "active": True,
            "reason": "role_aware_virtual_premaster_strategy_solved",
            "solver_input_basis": {
                "audio_basis": "post_pre_clean_audio_if_preclean_ran_else_original_analysis_audio",
                "analysis_basis": "post-clean/base fast QC plus Stage A role/profile/protection decision context",
                "premaster_policy": "do not build a separate premaster before the solver; virtual test selects the premaster correction strategy first",
            },
            "musical_role_identity_map": role_ctx,
            "representative_section_map": section_report,
            "virtual_limiter_sweep": {
                "method": "representative_sections_proxy_clip_limit",
                "sweep_gain_min_db": 0.0,
                "sweep_gain_max_db": round(max_push, 3),
                "sweep_step_db": round(step, 3),
                "user_defined_max_push_rule": False,
                "computational_guard_db": round(computational_guard_db, 3),
                "guard_policy": "internal crash guard only; strategic maximum is highest solvable gain before unresolved blockers",
                "proxy_oversample_factor": os_factor,
                "planner_ceiling_dbtp": round(planner_ceiling, 3),
                "proxy_clip_mix": round(clip_mix, 4),
                "proxy_clip_drive": round(clip_drive, 3),
                "rows_sample": sweep_rows,
            },
            "blocker_capacity_map": {
                "highest_clean_gain_db": round(highest_clean_gain, 3),
                "highest_solvable_gain_db": round(highest_solvable_gain, 3),
                "first_blocker": first_blocker,
                "capacity_by_gain": capacity_by_gain,
                "stop_condition": "no_solvable_gain_left_or_musical_damage_risk" if solution.get("stop_reasons") else "highest_solvable_gain_with_planned_corrections",
                "definition_of_maximum": "highest gain reached before the next blocker remained unsolvable after planned virtual corrections; not a user/env max-push setting",
            },
            "blocker_solution_plan": solution,
            "final_push_strategy": {
                "recommended_input_gain_db": round(recommended_gain, 3),
                "expected_lufs_after_push_est": round(float(expected_lufs), 3) if expected_lufs is not None else None,
                "target_lufs": round(target_lufs, 3),
                "commercial_floor_lufs": round(floor_lufs, 3),
                "recommended_true_peak_ceiling_db": round(recommended_ceiling, 3),
                "ceiling_delta_db": solution.get("ceiling_delta_db", 0.0),
                "clip_drive_delta_db": solution.get("clip_drive_delta_db", 0.0),
                "clip_mix_delta": solution.get("clip_mix_delta", 0.0),
                "recommended_candidate_gains_db": _candidate_gains_from_strategy(recommended_gain, max_push, needed_to_floor, needed_to_target),
                "render_policy": "generate the actual premaster once from the selected role-aware correction plan, then apply the selected limiter gain/ceiling; final QC is authoritative",
            },
            "llm_solution_judge": {
                "enabled": _env_bool("BUSY_VIRTUAL_HOT_LLM_JUDGE", False),
                "active": False,
                "reason": "not_called_in_v8_5_3_20; numeric blocker map is prepared for future GPT-mini structured-output judge",
            },
        })
    except Exception as exc:
        report.update({"active": False, "reason": "virtual_solver_error", "error_type": type(exc).__name__, "error": str(exc)[:500]})
    return report


def _candidate_gains_from_strategy(recommended: float, max_push: float, needed_floor: float, needed_target: float) -> list[float]:
    vals = [0.55, 0.95]
    if needed_floor > 0.18:
        vals.append(needed_floor)
    if needed_target > 0.18:
        vals.append(needed_target)
    if recommended > 0.18:
        vals.extend([recommended - 0.50, recommended - 0.25, recommended, recommended + 0.25])
    # max_push is an internal computational guard, not a candidate target.
    upper = max(0.18, min(float(max_push), max(float(recommended) + 0.50, float(needed_target) + 0.25, 1.0)))
    out = sorted({round(float(np.clip(v, 0.18, upper)), 3) for v in vals if v >= 0.18})
    out = sorted(out, key=lambda v: (abs(v - recommended), v))[: int(_env_float("BUSY_COMMERCIAL_FINISH_MAX_CANDIDATES", 6, 1, 8))]
    return sorted(out)
