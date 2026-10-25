# -*- coding: utf-8 -*-
"""
心电分析工具集

这里的每个函数都是智能体的一个"工具"：输入输出都是普通数据结构，
与有没有大模型无关，可以单独调用、单独测试。

分层：
  感知层  ecg_load / quality_check / preprocess / detect_beats
  测量层  basic_measures / hrv_metrics / interval_measures
  分类层  build_beat_features / classify_beats
  知识层  lookup_critical
  可视化  plot_result
"""
import os
import json
import re

import numpy as np
import wfdb
import neurokit2 as nk
from scipy.signal import welch, resample

# NumPy < 2.0 兼容：neurokit2 的 hrv_frequency 使用 np.trapezoid
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "models", "beat_classifier.pkl")
GUIDELINE_PATH = os.path.join(HERE, "guideline.md")

LEAD_NAMES = {0: "MLII", 1: "V5"}

# AAMI 标准把 MIT-BIH 的逐搏标注归并为五类
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",      # 正常与束支传导阻滞
    "A": "S", "a": "S", "J": "S", "S": "S",                # 室上性异位
    "V": "V", "E": "V",                                     # 室性异位
    "F": "F",                                               # 融合搏动
    "/": "Q", "f": "Q", "Q": "Q",                           # 未分类
}
AAMI_NAMES = {"N": "正常", "S": "室上性异位", "V": "室性异位", "F": "融合搏动", "Q": "未分类"}

# 记录 102/104/107/217 等首导联不是 MLII，这里统一标注
LEAD_OVERRIDE = {"102": "V5", "104": "V2", "107": "V5", "217": "MLII"}


# ============================================================
# 感知层
# ============================================================
def ecg_load(record="100", data_dir=DATA_DIR):
    """读入一条 MIT-BIH 记录，返回波形矩阵、采样率和导联名。"""
    rec = wfdb.rdrecord(os.path.join(data_dir, str(record)))
    return {
        "record": str(record),
        "signal": rec.p_signal,        # (采样点数, 导联数)，单位 mV
        "fs": float(rec.fs),
        "leads": list(rec.sig_name),
        "n_samples": int(rec.p_signal.shape[0]),
        "duration_min": round(rec.p_signal.shape[0] / float(rec.fs) / 60, 1),
    }


def _read_text_lines(source):
    """把路径或 file-like 读成文本行。"""
    if hasattr(source, "read"):
        text = source.read()
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
    else:
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    return text.splitlines()


def _parse_csv_meta(lines):
    """从厂商导出的 key,value 头里尽量读出采样率/导联/单位。"""
    meta = {}
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0]:
            meta[parts[0]] = parts[1]
    fs = None
    for key in ("采样率", "采样频率", "sampling rate", "fs"):
        if key in meta:
            m = re.search(r"([0-9]+(?:\.[0-9]+)?)", meta[key])
            if m:
                fs = float(m.group(1))
                break
    lead = meta.get("导联") or meta.get("lead") or meta.get("Lead")
    unit = meta.get("单位") or meta.get("unit") or meta.get("Unit")
    return {"fs": fs, "lead": lead, "unit": unit,
            "device": meta.get("设备") or meta.get("device"),
            "raw_meta": meta}


def _read_numeric_csv(source):
    """跳过厂商元数据头，读到第一行数值开始解析。

    支持单列/多列；多列时 v2.0 只取第一列（完整多导联适配留给 v2.1）。
    """
    lines = _read_text_lines(source)
    start = None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        first = s.split(",")[0].strip()
        try:
            float(first)
            start = i
            break
        except ValueError:
            continue
    if start is None:
        raise ValueError("文件里找不到数值数据行，请确认是心电采样值")
    rows = []
    for line in lines[start:]:
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    arr = np.asarray(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, _parse_csv_meta(lines)


def import_csv_ecg(source, fs, record_name, lead_name="CSV"):
    """把实验设备导出的 CSV 转成 WFDB 记录，之后就能当普通记录使用。

    同时保存一份**原始 ADC 计数** sidecar（`<record>_raw.npy` / `_raw.json`），
    供 Cardia 2.0 的上传路径使用。V2 的特征管线吃数字量，不能走 mV 缩放后的
    WFDB 记录（那会二次量化，量纲也对不上）。

    当前 v2.0 只把第一列当作单导联；多列/多厂商自动识别在 v2.1 ingest 层做。
    """
    arr, meta = _read_numeric_csv(source)
    n_cols = int(arr.shape[1])
    x = arr[:, 0] if n_cols > 0 else arr.ravel()
    if (not fs or float(fs) <= 0) and meta.get("fs"):
        fs = meta["fs"]                  # 厂商 CSV 头里有采样率时自动采用
    fs = float(fs)
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 100:
        raise ValueError("文件内容过少，请确认是单列的心电采样值")

    # ---- 原始 ADC / mV 数值 sidecar（V2 上传路径用）----
    raw = x.copy()
    np.save(os.path.join(DATA_DIR, f"{record_name}_raw.npy"), raw)
    with open(os.path.join(DATA_DIR, f"{record_name}_raw.json"), "w",
              encoding="utf-8") as f:
        json.dump({"fs": float(fs), "fs_from_file": meta.get("fs"),
                   "lead": meta.get("lead") or lead_name, "unit": meta.get("unit"),
                   "device": meta.get("device"), "source": "csv",
                   "n_samples": int(raw.size), "n_columns": n_cols,
                   "column_used": 0}, f, ensure_ascii=False, indent=2)

    x = x - np.median(x)                       # 去掉 ADC 直流偏置
    ref = float(np.percentile(np.abs(x), 99.5))
    if ref > 0:
        x = x / ref * 1.5

    wfdb.wrsamp(record_name, fs=float(fs), units=["mV"], sig_name=[lead_name],
                p_signal=x.reshape(-1, 1), fmt=["16"], write_dir=DATA_DIR)
    return {
        "record": record_name,
        "n_samples": int(x.size),
        "fs": float(fs),
        "duration_min": round(x.size / fs / 60, 2),
        "n_columns": n_cols,
        "fs_from_file": meta.get("fs"),
        "lead": meta.get("lead") or lead_name,
        "unit": meta.get("unit"),
        "device": meta.get("device"),
        "raw_sidecar": f"{record_name}_raw.npy",
    }


def quality_check(signal, fs):
    """检查导联脱落、信号饱和与高频噪声。"""
    issues = []
    for i in range(signal.shape[1]):
        lead = signal[:, i]
        if np.std(lead) < 1e-6:
            issues.append(f"导联{i}信号恒定，疑似脱落")
        elif np.max(np.abs(lead)) > 5.0:
            issues.append(f"导联{i}幅值过大，疑似伪差或饱和")

    freqs, psd = welch(signal[:, 0], fs=fs, nperseg=min(1024, len(signal)))
    hf_ratio = float(psd[freqs >= 40].sum() / psd.sum())
    if hf_ratio > 0.10:
        issues.append(f"高频噪声偏高（占比 {hf_ratio:.1%}）")

    return {"ok": len(issues) == 0, "issues": issues,
            "hf_noise_ratio": round(hf_ratio, 4)}


def preprocess(signal, fs, lead=0):
    """滤波去噪。目前用 neurokit2 的 ecg_clean（0.5–40 Hz 带通 + 陷波）。"""
    return nk.ecg_clean(signal[:, lead], sampling_rate=fs, method="neurokit")


def detect_beats(clean, fs):
    """定位 R 波，返回采样点下标。"""
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")
    return np.asarray(info["ECG_R_Peaks"])


# ============================================================
# 测量层
# ============================================================
def basic_measures(rpeaks, fs):
    """心率与 RR 间期的基础统计。"""
    rr = np.diff(rpeaks) / fs
    return {
        "n_beats": int(len(rpeaks)),
        "mean_hr_bpm": round(float(60.0 / rr.mean()), 1),
        "min_hr_bpm": round(float(60.0 / rr.max()), 1),
        "max_hr_bpm": round(float(60.0 / rr.min()), 1),
        "mean_rr_ms": round(float(rr.mean() * 1000), 1),
    }


def hrv_metrics(rpeaks, fs):
    """心率变异性：时域指标自算，频域指标交给 neurokit2。"""
    rr = np.diff(rpeaks) / fs * 1000.0          # ms
    diff_rr = np.diff(rr)
    out = {
        "sdnn_ms": round(float(rr.std(ddof=1)), 1),
        "rmssd_ms": round(float(np.sqrt(np.mean(diff_rr ** 2))), 1),
        "pnn50_pct": round(float(np.mean(np.abs(diff_rr) > 50) * 100), 1),
    }
    try:
        freq = nk.hrv_frequency(rpeaks, sampling_rate=fs, show=False,
                                normalize=False)
        def _finite(x, nd=1):
            x = float(x)
            return round(x, nd) if np.isfinite(x) else None
        lf = _finite(freq["HRV_LF"].iloc[0], 1)
        hf = _finite(freq["HRV_HF"].iloc[0], 1)
        ratio = _finite(freq["HRV_LFHF"].iloc[0], 2)
        out.update({"lf_ms2": lf, "hf_ms2": hf, "lf_hf_ratio": ratio})
        if lf is None or hf is None or ratio is None:
            out["freq_note"] = "记录过短，部分频域指标不可用"
    except Exception as e:
        out["freq_error"] = str(e)
    return out


def rhythm_summary(rpeaks, fs):
    """心律规整性：用 RR 间期的变异系数判断是否整齐。"""
    rr = np.diff(rpeaks) / fs
    cv = float(rr.std(ddof=1) / rr.mean())
    if cv < 0.10:
        verdict = "regular"
    elif cv < 0.20:
        verdict = "mildly_irregular"
    else:
        verdict = "irregular"
    return {"rr_cv": round(cv, 3), "rhythm": verdict}


# ============================================================
# 分类层
# ============================================================
PRE_S, POST_S = 0.30, 0.40        # 形态窗口：R 波前 0.3 s 到后 0.4 s
MORPH_LEN = 40                    # 重采样后的形态向量长度


def beat_features(sig, r, fs, rr_pre, rr_post, rr_local):
    """单个心搏的特征向量：RR 间期特征 + 波形形态特征。"""
    pre, post = int(PRE_S * fs), int(POST_S * fs)
    seg = sig[r - pre: r + post]
    if len(seg) < pre + post:
        return None

    # 形态向量：重采样到固定长度并归一化，消除个体幅度差异
    m = resample(seg, MORPH_LEN)
    m = (m - m.mean()) / (m.std() + 1e-8)

    # QRS 宽度估计：以 R 波幅度 20% 为阈值，量连续超阈值的宽度
    half = int(0.15 * fs)
    win = seg[pre - half: pre + half]
    thr = 0.2 * np.max(np.abs(win))
    above = np.abs(win) > thr
    width = float(above.sum()) / fs * 1000.0

    return np.concatenate([
        [rr_pre, rr_post, rr_pre / rr_local, rr_post / rr_local,
         rr_local, width,
         seg.max() - seg.min(),                 # 峰峰值
         np.abs(seg).sum() / len(seg),          # 平均绝对幅值
         m[pre * MORPH_LEN // (pre + post)],    # R 波处的相对位置幅值
         ],
        m,
    ])


def build_beat_features(sig, rpeaks, fs):
    """对整条记录逐搏提特征，返回特征矩阵、有效心搏下标。"""
    rr = np.diff(rpeaks) / fs
    X, keep = [], []
    for i in range(2, len(rpeaks) - 1):
        r = rpeaks[i]
        if r - int(PRE_S * fs) < 0 or r + int(POST_S * fs) >= len(sig):
            continue
        rr_local = float(np.median(rr[max(0, i - 10):i + 10]))
        f = beat_features(sig, r, fs, float(rr[i - 1]), float(rr[i]), rr_local)
        if f is not None:
            X.append(f)
            keep.append(i)
    return np.asarray(X), np.asarray(keep)


def _load_model():
    import pickle
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def classify_beats_rf(record="100", data_dir=DATA_DIR):
    """v1 随机森林分类器：自动检测 R 峰 + 49 维特征 + RandomForest。"""
    data = ecg_load(record, data_dir)
    clean = preprocess(data["signal"], data["fs"])
    rpeaks = detect_beats(clean, data["fs"])
    X, keep = build_beat_features(clean, rpeaks, data["fs"])
    if len(X) == 0:
        return {
            "record": str(record),
            "n_beats": 0,
            "counts": {"N": 0, "S": 0, "V": 0, "F": 0},
            "abnormal_pct": None,
            "error": "没有足够的有效心搏用于随机森林分类",
        }

    model = _load_model()
    pred = model.predict(X)

    counts = {k: int((pred == k).sum()) for k in ("N", "S", "V", "F")}
    total = sum(counts.values()) or 1
    return {
        "record": str(record),
        "n_beats": int(total),
        "counts": counts,
        "abnormal_pct": round((total - counts["N"]) / total * 100, 2),
    }


def classify_beats_v1(record="100", data_dir=DATA_DIR):
    """v1 结果补齐成与 v2 一致的字段，方便界面并排比较。"""
    r = classify_beats_rf(record, data_dir)
    total = r["n_beats"]
    r.update({
        "engine": "fallback",
        "model": "RandomForest",
        "model_version": "1.0",
        "n_used": total,
        "n_chunks": None,
        "dropped_tail": 0,
        "mean_confidence": None,
        "class_mean_prob": None,
        "annotation_source": "detected",
        "experimental": False,
        "lead": "auto",
        "warnings": ["v1 使用 neurokit2 自动检测 R 峰；分类器为随机森林"],
        "inference_ms": None,
    })
    return r


def classify_beats(record="100", data_dir=DATA_DIR, engine="auto", polarity="auto"):
    """逐搏分类分发。

    engine:
        "auto" —— 优先 Cardia 2.0；V2 无法运行（如记录太短、无有效心搏）时回退 v1
        "v2"   —— 只用 Cardia 2.0
        "v1"   —— 只用 v1 随机森林
    polarity:
        仅 V2 上传路径使用，"auto" / "normal" / "inverted"
    """
    engine = (engine or "auto").lower()
    if engine in ("v2", "relay", "auto"):
        try:
            import cardia_classifier
            r = cardia_classifier.classify_record(record, data_dir=data_dir, polarity=polarity)
        except Exception as e:
            r = {"record": str(record), "error": f"{type(e).__name__}: {e}"}
        if "error" not in r:
            return r
        if engine in ("v2", "relay"):
            return r
        # auto: V2 跑不了，回退 v1，并保留原因
        v1 = classify_beats_v1(record, data_dir)
        v1["fallback_from"] = "cardia"
        v1["cardia_error"] = r.get("error")
        v1["warnings"].append(f"Cardia 2.0 未运行，已回退随机森林：{r.get('error')}")
        return v1

    return classify_beats_v1(record, data_dir)


def align_labels(data_dir, record, rpeaks, fs, tol=0.05):
    """把专家标注对齐到检出的 R 波上。

    标注位置与检测位置不完全重合，取容差 tol 秒内最近的标注作为该心搏的标签；
    对不上的心搏记为 None。
    """
    ann = wfdb.rdann(os.path.join(data_dir, str(record)), "atr")
    pos = np.asarray(ann.sample)
    sym = np.asarray(ann.symbol)
    keep = np.array([s in AAMI_MAP for s in sym])
    pos, sym = pos[keep], sym[keep]

    order = np.argsort(pos)
    pos, sym = pos[order], sym[order]

    tol_n = tol * fs
    j = np.searchsorted(pos, rpeaks)
    labels = []
    for i, r in enumerate(rpeaks):
        best = None
        for cand in (j[i] - 1, j[i]):
            if 0 <= cand < len(pos) and abs(pos[cand] - r) <= tol_n:
                best = AAMI_MAP[sym[cand]]
                break
        labels.append(best)
    return np.array(labels, dtype=object)


def collect_dataset(records, data_dir=DATA_DIR, verbose=False):
    """遍历给定记录，返回特征矩阵与标签。"""
    X_all, y_all = [], []
    for r in records:
        data = ecg_load(r, data_dir)
        clean = preprocess(data["signal"], data["fs"])
        rpeaks = detect_beats(clean, data["fs"])
        labels = align_labels(data_dir, r, rpeaks, data["fs"])
        X, keep = build_beat_features(clean, rpeaks, data["fs"])
        y = labels[keep]
        mask = np.array([v in ("N", "S", "V", "F") for v in y])
        X_all.append(X[mask])
        y_all.append(y[mask])
        if verbose:
            print(f"  {r}: 心搏 {len(X)}，有效标签 {int(mask.sum())}，"
                  f"时长 {data['duration_min']} 分钟")
    return np.vstack(X_all), np.concatenate(y_all)


def evaluate(model, X, y, title="评测结果"):
    """输出分类报告与混淆矩阵。"""
    from sklearn.metrics import classification_report, confusion_matrix
    labels = ["N", "S", "V", "F"]
    pred = model.predict(X)
    print(f"\n===== {title} =====")
    print(classification_report(y, pred, labels=labels, digits=3,
                                zero_division=0,
                                target_names=[f"{k}({AAMI_NAMES[k]})" for k in labels]))
    print("混淆矩阵（行=真实，列=预测，顺序 N/S/V/F）：")
    print(confusion_matrix(y, pred, labels=labels))
    return pred


# ============================================================
# 知识层
# ============================================================
def lookup_critical(conditions):
    """查询心电图危急值指南条目。

    conditions 可以是一个关键词字符串，也可以是若干关键词的列表。
    返回命中的条目文本。
    """
    if isinstance(conditions, str):
        conditions = [conditions]
    if not os.path.exists(GUIDELINE_PATH):
        return {"hits": [], "note": "guideline file not found"}

    hits = []
    with open(GUIDELINE_PATH, encoding="utf-8") as f:
        blocks = [b.strip() for b in f.read().split("\n---\n") if b.strip()]
    for block in blocks:
        if any(c and c in block for c in conditions):
            hits.append(block)
    return {"hits": hits, "n_hits": len(hits)}


# ============================================================
# 可视化
# ============================================================
def _pick_cjk_font():
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ["PingFang SC", "Songti SC", "Heiti SC", "Arial Unicode MS", "SimHei"]:
        if name in available:
            return name
    return None


def plot_result(signal, clean, rpeaks, fs, out_png, lead=0, seconds=10):
    """画 10 秒波形（标出 R 波）与 RR 间期序列。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    font = _pick_cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font]
    plt.rcParams["axes.unicode_minus"] = False

    n = int(seconds * fs)
    t = np.arange(n) / fs
    seg_peaks = rpeaks[rpeaks < n]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6))
    axes[0].plot(t, signal[:n, lead], color="0.75", lw=0.8, label="原始信号")
    axes[0].plot(t, clean[:n], color="#c0392b", lw=1.0, label="滤波后")
    axes[0].plot(seg_peaks / fs, clean[seg_peaks], "o", mfc="none",
                 mec="#2c3e50", ms=8, label="检测到的 R 波")
    axes[0].set_xlabel("时间 (s)"); axes[0].set_ylabel("幅值 (mV)")
    axes[0].set_title(f"首导联（{LEAD_OVERRIDE.get(str(lead), LEAD_NAMES.get(lead, lead))}）："
                      f"前 {seconds} 秒波形与 R 波检测")
    axes[0].legend(loc="upper right", fontsize=9); axes[0].grid(alpha=0.3)

    rr_ms = np.diff(rpeaks) / fs * 1000
    axes[1].plot(np.arange(1, len(rr_ms) + 1), rr_ms, "-o",
                 color="#2980b9", lw=0.9, ms=3)
    axes[1].axhline(rr_ms.mean(), color="0.5", ls="--", lw=0.9,
                    label=f"平均 {rr_ms.mean():.0f} ms")
    axes[1].set_xlabel("心搏序号"); axes[1].set_ylabel("RR 间期 (ms)")
    axes[1].set_title("RR 间期序列"); axes[1].legend(loc="upper right", fontsize=9)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
