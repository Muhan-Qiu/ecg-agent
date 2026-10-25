# -*- coding: utf-8 -*-
"""
Cardia 2.0 心搏分类器

职责：
  原始 ECG（MLII 数字量）+ R 峰标注/自动检测
    -> feature_extract.py 的 104 维特征
    -> preprocess.json 的 scaler / dt_median
    -> 每记录 64 beat 非重叠切块
    -> ONNX / TorchScript 前向
    -> 每 beat 4 类概率，汇总成工具返回值

只加载发布权重（ONNX / TorchScript），不包含训练代码或模型类。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import feature_extract as fe

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models" / "cardia"
DATA_DIR = HERE / "data"

MODEL_NAME = "Cardia 2.0"
MODEL_VERSION = "2.0"
FS_TARGET = 360
TARGET_P99 = 361.0          # MIT-BIH MLII 数字量的 p99.5 中位数，用于上传幅度校准
RAW_SUFFIX = "_raw.npy"
RAW_META_SUFFIX = "_raw.json"


# ------------------------------------------------------------
# 权重 / 预处理
# ------------------------------------------------------------
def load_preprocess(model_dir=None):
    model_dir = Path(model_dir or MODEL_DIR)
    with open(model_dir / "preprocess.json", "r", encoding="utf-8") as f:
        p = json.load(f)
    p["sequence_length"] = int(p.get("sequence_length", 64))
    p["stride"] = int(p.get("stride", p["sequence_length"]))
    return p


_MODEL_CACHE = {}


def _default_fmt():
    """有 torch 就用 TorchScript（加载快）；否则用 ONNX。

    注意：当前 ONNX 图有约 2.4 万个节点，ONNX Runtime 首次建会话在部分
    Windows 机器上要 ~1 分钟；TorchScript 加载约 2 秒。因此 App 默认 TS。
    """
    try:
        import torch  # noqa: F401
        return "ts"
    except Exception:
        return "onnx"


def load_model(model_dir=None, fmt=None):
    model_dir = Path(model_dir or MODEL_DIR)
    fmt = (fmt or _default_fmt()).lower()
    key = (str(model_dir), fmt)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if fmt == "onnx":
        import onnxruntime as ort
        path = model_dir / "cardia_k4_best.onnx"
        with open(path, "rb") as f:
            blob = f.read()
        model = ort.InferenceSession(blob, providers=["CPUExecutionProvider"])
    elif fmt == "ts":
        import torch
        path = model_dir / "cardia_k4_best_ts.pt"
        try:
            with open(path, "rb") as f:
                model = torch.jit.load(f, map_location="cpu")
        except Exception:
            # Windows/中文路径下 torch.jit.load(path) 可能失败；退到临时 ASCII 路径
            import shutil
            import tempfile
            tmp = os.path.join(tempfile.gettempdir(), "cardia_k4_release_ts.pt")
            shutil.copyfile(path, tmp)
            model = torch.jit.load(tmp, map_location="cpu")
        model.eval()
    else:
        raise ValueError(f"unknown format: {fmt}")
    _MODEL_CACHE[key] = model
    return model


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _forward(model, fmt, x, dt):
    if fmt == "onnx":
        out = model.run(None, {"x": x.astype(np.float32), "dt": dt.astype(np.float32)})
        return out[0]
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(x.astype(np.float32)),
                    torch.from_numpy(dt.astype(np.float32)))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().cpu().numpy()


def _chunk_starts(n, T, stride):
    return list(range(0, n - T + 1, stride))


def predict_features(features, pre_r, pp, model, fmt=None, pad_short=True):
    """features [n,104] 未标准化；pre_r [n] 原始采样点。

    返回 probs [n_chunks, T, 4] 或短记录 padding 后的 [1, n, 4]。
    短记录（n < T）用零特征 + dt=1.0 填充到 T，只返回真实 n 个 beat 的概率。
    """
    T = pp["sequence_length"]
    stride = pp["stride"]
    mean = np.asarray(pp["scaler_mean"], dtype=np.float64)
    scale = np.asarray(pp["scaler_scale"], dtype=np.float64)
    x_all = (np.asarray(features, dtype=np.float64) - mean) / scale
    dt_all = np.asarray(pre_r, dtype=np.float64) / float(pp["dt_median"])

    n = len(x_all)
    if n < T:
        if not pad_short:
            return np.zeros((0, 0, 4), dtype=np.float64), []
        xpad = np.zeros((1, T, x_all.shape[1]), dtype=np.float64)
        dpad = np.ones((1, T), dtype=np.float64)
        xpad[0, :n] = x_all
        dpad[0, :n] = dt_all
        logits = _forward(model, fmt, xpad, dpad)
        return _softmax(logits)[:, :n, :], [0]

    starts = _chunk_starts(n, T, stride)
    if not starts:
        return np.zeros((0, 0, 4), dtype=np.float64), []

    xs = np.stack([x_all[s:s + T] for s in starts], axis=0)
    dts = np.stack([dt_all[s:s + T] for s in starts], axis=0)
    try:
        logits = _forward(model, fmt, xs, dts)
    except Exception:
        # 某些 TorchScript/ONNX 导出只接受 batch=1，退化为逐块前向
        outs = [_forward(model, fmt, xs[i:i + 1], dts[i:i + 1])[0]
                for i in range(len(starts))]
        logits = np.stack(outs, axis=0)
    return _softmax(logits), starts


# ------------------------------------------------------------
# 读入：内置 MIT-BIH / 上传记录
# ------------------------------------------------------------
def _read_wfdb_digital(record, data_dir=None):
    """返回 (数字量 MLII, fs, annotations 或 None)。"""
    import wfdb
    prefix = str(Path(data_dir or DATA_DIR) / str(record))
    rec = wfdb.rdrecord(prefix, physical=False)
    names = [n.upper() for n in rec.sig_name]
    ch = names.index("MLII") if "MLII" in names else 0
    sig = rec.d_signal[:, ch].astype(np.float64)
    fs = float(rec.fs)
    anns = None
    try:
        ann = wfdb.rdann(prefix, "atr")
        anns = list(zip([int(s) for s in ann.sample], [str(s) for s in ann.symbol]))
    except Exception:
        pass
    return sig, fs, anns, rec.sig_name[ch] if rec.sig_name else "unknown"


def _resample_to_360(x, fs):
    if abs(fs - FS_TARGET) < 1e-6:
        return x, fs
    from math import gcd
    from scipy.signal import resample_poly
    fs = float(fs)
    g = gcd(int(round(FS_TARGET)), int(round(fs)))
    up, down = int(round(FS_TARGET)) // g, int(round(fs)) // g
    return resample_poly(x, up, down), float(FS_TARGET)


def _calibrate_amplitude(x, target_p99=TARGET_P99):
    """把上传信号的鲁棒幅度校准到 MIT-BIH MLII 数字量量级。

    只校准幅度，不改变时间轴；绝对电压仍然不可信。
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - np.median(x)
    ref = float(np.percentile(np.abs(x), 99.5))
    if ref <= 0:
        return x, False
    return x * (target_p99 / ref), True


def _maybe_invert(x, peaks, polarity="auto"):
    """单导联上传的极性启发式：QRS 主波向下时翻转。"""
    if polarity == "inverted":
        return -x, True
    if polarity == "normal":
        return x, False
    if len(peaks) < 5:
        return x, False
    scores = []
    for p in peaks:
        lo, hi = int(p) - 40, int(p) + 40
        if lo < 0 or hi > len(x):
            continue
        beat = x[lo:hi] - np.median(x[lo:hi])
        scores.append(float(beat.max() + beat.min()))
    if not scores:
        return x, False
    return (-x, True) if np.median(scores) < 0 else (x, False)


def _detect_peaks(x, fs):
    """自动检测 R 峰；用 ecg_tools 的 neurokit2 管线，避免重复实现。"""
    import ecg_tools as T
    phys = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    phys = phys / 200.0  # 近似 mV；检测只关心相对幅度
    clean = T.preprocess(phys, fs)
    return np.asarray(T.detect_beats(clean, fs))


def _load_upload_raw(record, data_dir=None):
    """读取 import_csv_ecg 保存的原始 ADC 计数 sidecar。

    返回 (raw_counts, fs, warnings) 或 None。
    """
    d = Path(data_dir or DATA_DIR)
    raw_path = d / f"{record}{RAW_SUFFIX}"
    if not raw_path.exists():
        return None
    x = np.load(raw_path).astype(np.float64).ravel()
    meta_path = d / f"{record}{RAW_META_SUFFIX}"
    fs = 500.0
    lead = None
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            # 厂商头里的采样率优先于用户手填值（V2 路径）
            fs = float(meta.get("fs_from_file") or meta.get("fs", fs))
            lead = meta.get("lead")
        except Exception:
            pass
    return x, fs, [f"检测到上传原始 ADC 计数 sidecar：{raw_path.name}"], lead


# ------------------------------------------------------------
# 主入口
# ------------------------------------------------------------
def classify_record(record, data_dir=None, fmt=None, polarity="auto",
                    allow_detected=True, model_dir=None):
    """对一条记录做 Cardia 2.0 逐搏分类，返回可直接给 Agent 的汇总 JSON。"""
    t0 = time.perf_counter()
    fmt = (fmt or _default_fmt()).lower()
    warnings = []
    experimental = False

    upload = _load_upload_raw(record, data_dir)
    if upload is not None:
        raw, fs_in, w, file_lead = upload
        warnings.extend(w)
        experimental = True
        if len(raw) < 200:
            return _error(record, "记录过短，无法提取 104 维特征")
        x, fs = _resample_to_360(raw, fs_in)
        if fs_in != FS_TARGET:
            warnings.append(f"采样率 {fs_in:g} Hz 已重采样到 {FS_TARGET} Hz")
        x, calibrated = _calibrate_amplitude(x)
        if calibrated:
            warnings.append("幅度已按 p99.5 校准到 MIT-BIH 数字量级；绝对电压不可信")
        sig, annotations, lead = x, None, (file_lead or "upload")
        annotation_source = "detected"
    else:
        sig, fs, annotations, lead = _read_wfdb_digital(record, data_dir)
        if annotations:
            annotation_source = "wfdb_atr"
        else:
            annotation_source = "detected"
            experimental = True
            warnings.append("该记录没有 .atr 专家标注，使用自动检测的 R 峰")

    if abs(fs - FS_TARGET) > 1e-6:
        sig, fs = _resample_to_360(sig, fs)
        warnings.append(f"采样率 {fs:g} Hz 已重采样到 {FS_TARGET} Hz")

    if annotation_source == "detected":
        if not allow_detected:
            return _error(record, "该记录没有专家 R 峰标注，且允许自动检测已关闭")
        peaks = _detect_peaks(sig, fs)
        if len(peaks) == 0:
            return _error(record, "未检测到 R 峰，无法分类")
        sig, inverted = _maybe_invert(sig, peaks, polarity)
        if inverted:
            warnings.append("QRS 主波向下，已自动翻转极性；如结果异常请手动指定 polarity")
        annotations = [(int(p), "N") for p in peaks]
        warnings.append(f"使用自动检测的 {len(peaks)} 个 R 峰（无专家标注）")

    if lead.upper() != "MLII":
        warnings.append(f"当前导联为 {lead}；V2 在 MIT-BIH MLII 上训练，形态特征对导联敏感")

    # 104 维特征 + 前向
    features, pre_r, _ = fe.extract_104(sig, annotations)
    n_beats = int(len(features))
    if n_beats == 0:
        msg = "没有有效心搏（窗口越界或符号不在 AAMI 四类内）"
        if annotation_source == "detected":
            msg += "；请检查采样率、导联和信号质量"
        return _error(record, msg)

    pp = load_preprocess(model_dir)
    model = load_model(model_dir, fmt)
    probs, starts = predict_features(features, pre_r, pp, model, fmt)
    T = pp["sequence_length"]
    n_used = int(probs.reshape(-1, probs.shape[-1]).shape[0]) if probs.size else 0
    n_chunks = int(probs.shape[0]) if probs.size else 0
    dropped_tail = int(n_beats - n_used)
    padded = bool(n_beats < T and n_used > 0)
    if padded:
        warnings.append(
            f"记录只有 {n_beats} 个有效心搏（<64），已零填充到 64 搏；短记录结果仅供参考")
    if dropped_tail and not padded:
        warnings.append(f"末尾 {dropped_tail} 个心搏不足 64 beat chunk，已丢弃")

    if n_used == 0:
        return _error(
            record,
            f"有效心搏只有 {n_beats} 个，无法组成预测输入",
            extra={"n_beats": n_beats, "annotation_source": annotation_source,
                   "experimental": experimental, "warnings": warnings},
        )

    names = pp["class_order"]
    flat = probs.reshape(-1, probs.shape[-1])
    pred = np.argmax(flat, axis=-1)
    counts = {names[k]: int((pred == k).sum()) for k in range(len(names))}
    total = sum(counts.values()) or 1
    mean_prob = flat.mean(axis=0)
    mean_conf = float(flat.max(axis=-1).mean())

    return {
        "record": str(record),
        "engine": "v2",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "n_beats": n_beats,
        "n_used": n_used,
        "n_chunks": n_chunks,
        "n_padded": int(T - n_beats) if padded else 0,
        "dropped_tail": dropped_tail,
        "padded": padded,
        "counts": counts,
        "abnormal_pct": round((total - counts.get("N", 0)) / total * 100, 2),
        "mean_confidence": round(mean_conf, 4),
        "class_mean_prob": {names[k]: round(float(mean_prob[k]), 4)
                            for k in range(len(names))},
        "annotation_source": annotation_source,
        "experimental": bool(experimental),
        "lead": lead,
        "runtime": fmt,
        "warnings": warnings,
        "inference_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def _error(record, message, extra=None):
    out = {
        "record": str(record),
        "engine": "v2",
        "model": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "error": str(message),
        "counts": {"N": 0, "SVEB": 0, "VEB": 0, "F": 0},
        "abnormal_pct": None,
        "annotation_source": None,
        "experimental": True,
        "warnings": [str(message)],
    }
    if extra:
        out.update(extra)
    return out


# ------------------------------------------------------------
# 模型卡（供 model_info 工具使用；数字来自 DS2 发布评测）
# ------------------------------------------------------------
def model_info():
    return {
        "name": MODEL_NAME,
        "version": MODEL_VERSION,
        "type": "self-developed ECG sequence classifier (Cardia 2.0)",
        "params": 809332,
        "input": "104-dim beat features, 64-beat non-overlapping chunks",
        "features": "RR 8 + u-LBP 59 + wavelet 23 + HOS 10 + myMorph 4",
        "training_data": "MIT-BIH Arrhythmia Database DS1 (22 records), AAMI 4-class",
        "eval_data": "MIT-BIH DS2 (22 records), inter-patient; DS2 evaluated once",
        "metrics_ds2": {
            "score": 74.00,
            "score_max": 100,
            "n_beats": 49024,
        },
        "deployed_checkpoint_note": "Released checkpoint score = 74.00 / 100; "
                                    "the same config's 5-seed mean = 70.02 / 100.",
        "limitations": [
            "Research/education only; not a medical device.",
            "The composite score is an overall metric and does not replace per-class clinical validation.",
            "Trained on MIT-BIH MLII at 360 Hz; other leads/sampling rates are out of distribution.",
            "Single dataset and single inter-patient split; no external validation.",
            "The 104-dim feature pipeline is a reimplementation of a GPL-3.0 public pipeline; "
            "the Cardia 2.0 model, training and selection are self-developed.",
        ],
    }
