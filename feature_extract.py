# -*- coding: utf-8 -*-
"""
feature_extract.py —— 原始 ECG -> 他们的 104 维特征（独立推理用）

来源与边界
----------
按 mondejar/ecg-classification 的公开管线复刻（MLII、两次中值滤波、
R 峰 ±20 局部最大、±90 窗、RR4 + normRR4 + u-lbp59 + wvlt23 + HOS10 + myMorph4）。
**不包含** Relay 模型/训练代码；只做特征提取。

输入：
  - WFDB 记录：`--record path/to/100`（读 .hea/.dat/.atr，取 MLII，数字量）
  - 或他们格式的 CSV+TXT：`--csv 100.csv --txt 100.txt`

输出：
  features [n,104]（未标准化；列顺序见 FEATURE_NAMES）
  pre_R   [n]（原始采样点；features[:,0]）
  labels  [n]（0=N,1=SVEB,2=VEB,3=F；可选）
"""
import os
import numpy as np
import operator
import scipy.stats
import pywt
from scipy.signal import medfilt

WINL = WINR = 90
SIZE_RR_MAX = 20
FS = 360

MITBIH_CLASSES = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']
AAMI_CLASSES = [['N', 'L', 'R'], ['A', 'a', 'J', 'S', 'e', 'j'],
                ['V', 'E'], ['F']]

FEATURE_NAMES = (
    ["pre_R", "post_R", "local_R", "global_R",
     "norm_pre_R", "norm_post_R", "norm_local_R", "norm_global_R"]
    + [f"u-lbp_{i:02d}" for i in range(59)]
    + [f"wvlt_{i:02d}" for i in range(23)]
    + [f"HOS_{i:02d}" for i in range(10)]
    + [f"myMorph_{i:02d}" for i in range(4)]
)


class RRIntervals:
    def __init__(self):
        self.pre_R = np.array([])
        self.post_R = np.array([])
        self.local_R = np.array([])
        self.global_R = np.array([])


def compute_RR_intervals(R_poses):
    """与 features_ECG.compute_RR_intervals 相同。"""
    f = RRIntervals()
    R = np.asarray(R_poses, dtype=float).ravel()
    if len(R) == 0:
        return f
    if len(R) == 1:
        f.pre_R = np.array([0.0]); f.post_R = np.array([0.0])
        f.local_R = np.array([0.0]); f.global_R = np.array([0.0])
        return f

    pre_R = np.array([0.0])
    post_R = np.array([R[1] - R[0]])
    for i in range(1, len(R) - 1):
        pre_R = np.append(pre_R, R[i] - R[i - 1])
        post_R = np.append(post_R, R[i + 1] - R[i])
    pre_R[0] = pre_R[1] if len(pre_R) > 1 else 0.0
    pre_R = np.append(pre_R, R[-1] - R[-2])
    post_R = np.append(post_R, post_R[-1])

    local_R = np.array([])
    for i in range(len(R)):
        num, avg_val = 0, 0.0
        for j in range(-9, 1):
            if j + i >= 0:
                avg_val += pre_R[i + j]
                num += 1
        local_R = np.append(local_R, avg_val / float(num))

    global_R = np.array([pre_R[0]])
    for i in range(1, len(R)):
        num, avg_val = 0, 0.0
        for j in range(0, i):
            if (R[i] - R[j]) < 108000:
                avg_val += pre_R[j]
                num += 1
        global_R = np.append(global_R, avg_val / float(max(num, 1)))

    f.pre_R, f.post_R, f.local_R, f.global_R = pre_R, post_R, local_R, global_R
    return f


def compute_wavelet_descriptor(beat, family="db1", level=3):
    coeffs = pywt.wavedec(beat, pywt.Wavelet(family), level=level)
    return coeffs[0]


def compute_my_own_descriptor(beat, winL=90, winR=90):
    R_pos = int((winL + winR) / 2)
    R_value = beat[R_pos]
    my_morph = np.zeros(4)
    x_values = np.zeros(4)
    y_values = np.zeros(4)
    x_values[0], y_values[0] = max(enumerate(beat[0:40]), key=operator.itemgetter(1))
    x_values[1], y_values[1] = min(enumerate(beat[75:85]), key=operator.itemgetter(1))
    x_values[2], y_values[2] = min(enumerate(beat[95:105]), key=operator.itemgetter(1))
    x_values[3], y_values[3] = max(enumerate(beat[150:180]), key=operator.itemgetter(1))
    x_values[1] += 75; x_values[2] += 95; x_values[3] += 150

    x_max, x_min = max(x_values), min(x_values)
    y_max = max(np.append(y_values, R_value))
    y_min = min(np.append(y_values, R_value))
    R_pos_n = (R_pos - x_min) / (x_max - x_min)
    R_value_n = (R_value - y_min) / (y_max - y_min)
    for n in range(4):
        x_values[n] = (x_values[n] - x_min) / (x_max - x_min)
        y_values[n] = (y_values[n] - y_min) / (y_max - y_min)
        my_morph[n] = np.linalg.norm([R_pos_n - x_values[n], R_value_n - y_values[n]])
    my_morph[np.isnan(my_morph)] = 0.0
    return my_morph


def compute_hos_descriptor(beat, n_intervals=6, lag=30):
    hos_b = np.zeros((n_intervals - 1) * 2)
    for i in range(n_intervals - 1):
        pose = lag * (i + 1)
        interval = beat[(pose - (lag // 2)):(pose + (lag // 2))]
        hos_b[i] = scipy.stats.skew(interval, 0, True)
        if np.isnan(hos_b[i]):
            hos_b[i] = 0.0
        hos_b[(n_intervals - 1) + i] = scipy.stats.kurtosis(interval, 0, False, True)
        if np.isnan(hos_b[(n_intervals - 1) + i]):
            hos_b[(n_intervals - 1) + i] = 0.0
    return hos_b


def compute_hos_descriptor_batch(beats, n_intervals=6, lag=30):
    """compute_hos_descriptor 的批量版；与 scipy 口径一致。

    注意原代码调用的是 kurtosis(interval, 0, False, True)：
    fisher=False（原始峰度，不是超额峰度）、bias=True。
    """
    beats = np.asarray(beats, dtype=np.float64)
    if beats.ndim != 2:
        raise ValueError("beats must be [n_beats, length]")
    half = lag // 2
    intervals = []
    for i in range(n_intervals - 1):
        pose = lag * (i + 1)
        intervals.append(beats[:, pose - half:pose + half])
    x = np.stack(intervals, axis=1)          # [n, n_intervals-1, lag]
    d = x - x.mean(axis=-1, keepdims=True)
    var = (d ** 2).mean(axis=-1)
    std = np.sqrt(var)
    skew = np.zeros_like(var)
    m1 = std > 0
    skew[m1] = (d[m1] ** 3).mean(axis=-1) / (std[m1] ** 3)
    kurt = np.zeros_like(var)
    m2 = var > 0
    kurt[m2] = (d[m2] ** 4).mean(axis=-1) / (var[m2] ** 2)
    return np.concatenate([skew, kurt], axis=1)


UNIFORM_PATTERN_LIST = np.array([
    0, 1, 2, 3, 4, 6, 7, 8, 12, 14, 15, 16, 24, 28, 30, 31, 32, 48, 56, 60,
    62, 63, 64, 96, 112, 120, 124, 126, 127, 128, 129, 131, 135, 143, 159,
    191, 192, 193, 195, 199, 207, 223, 224, 225, 227, 231, 239, 240, 241,
    243, 247, 248, 249, 251, 252, 253, 254, 255])


def compute_Uniform_LBP(signal, neigh=8):
    hist_u_lbp = np.zeros(59, dtype=float)
    for i in range(neigh // 2, len(signal) - neigh // 2):
        pattern = np.zeros(neigh)
        ind = 0
        for n in list(range(-neigh // 2, 0)) + list(range(1, neigh // 2 + 1)):
            if signal[i] > signal[i + n]:
                pattern[ind] = 1
            ind += 1
        pattern_id = int("".join(str(c) for c in pattern.astype(int)), 2)
        if pattern_id in UNIFORM_PATTERN_LIST:
            pattern_uniform_id = int(np.argwhere(UNIFORM_PATTERN_LIST == pattern_id)[0][0])
        else:
            pattern_uniform_id = 58
        hist_u_lbp[pattern_uniform_id] += 1.0
    return hist_u_lbp


# 256 个 8-bit pattern -> 59 个 uniform bin 的查找表（58 = non-uniform）
_UNIFORM_LUT = np.full(256, 58, dtype=np.int64)
for _i, _p in enumerate(UNIFORM_PATTERN_LIST):
    _UNIFORM_LUT[int(_p)] = _i


def compute_Uniform_LBP_batch(beats, neigh=8):
    """compute_Uniform_LBP 的批量版；逐元素结果与单搏版一致。"""
    beats = np.asarray(beats, dtype=np.float64)
    if beats.ndim != 2:
        raise ValueError("beats must be [n_beats, length]")
    half = neigh // 2
    core = beats[:, half:beats.shape[1] - half]
    pattern = np.zeros(core.shape, dtype=np.int64)
    offsets = list(range(-half, 0)) + list(range(1, half + 1))
    for bit, off in enumerate(offsets):
        nb = beats[:, half + off:beats.shape[1] - half + off]
        pattern |= (core > nb).astype(np.int64) << (neigh - 1 - bit)
    ids = _UNIFORM_LUT[pattern]
    hist = np.zeros((len(beats), 59), dtype=np.float64)
    for k in range(59):
        hist[:, k] = (ids == k).sum(axis=1)
    return hist


# --------------------------------------------------------------------------
# 读数据
# --------------------------------------------------------------------------
def read_csv_record(csv_path, txt_path, rec_num=None):
    """他们格式：CSV 第 1 行列头，之后 `序号, 通道0, 通道1`；TXT 第 1 行列头，之后 `序号 位置 符号`。"""
    if rec_num is None:
        rec_num = int(str(os.path.basename(csv_path))[:3]) if "os" in globals() else 0
    mlii_idx, v1_idx = 1, 2
    if int(rec_num) == 114:
        mlii_idx, v1_idx = 2, 1
    mlii, v1 = [], []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            row = line.strip().split(",")
            if len(row) < 3:
                continue
            mlii.append(int(float(row[mlii_idx])))
            v1.append(int(float(row[v1_idx])))
    anns = []
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                anns.append((int(parts[1]), parts[2]))
    return mlii, v1, anns


def read_wfdb_record(prefix, ann_ext="atr"):
    """WFDB 记录：取 MLII 通道的数字量 + atr 标注。"""
    import wfdb
    # physical=False 才返回 d_signal；特征管线用的是数字量（与 CSV 一致）
    rec = wfdb.rdrecord(prefix, physical=False)
    names = [n.upper() for n in rec.sig_name]
    if "MLII" in names:
        ch = names.index("MLII")
    else:
        ch = 0
    sig = rec.d_signal[:, ch].astype(np.int64)
    ann = wfdb.rdann(prefix, ann_ext)
    anns = list(zip([int(s) for s in ann.sample], [str(s) for s in ann.symbol]))
    return sig.tolist(), None, anns


# --------------------------------------------------------------------------
# 组 beat + 104 维
# --------------------------------------------------------------------------
def build_beats(mlii, anns):
    mlii = np.asarray(mlii, dtype=np.float64)
    # 两次中值滤波去基线（与他们的 do_preprocess=True 一致）
    baseline = medfilt(mlii, 71)
    baseline = medfilt(baseline, 215)
    mlii = mlii - baseline

    beats, labels, valid, R_pos = [], [], [], []
    for pos0, sym in anns:
        pos = int(pos0)
        if pos > SIZE_RR_MAX and pos < (len(mlii) - SIZE_RR_MAX):
            idx, _ = max(enumerate(mlii[pos - SIZE_RR_MAX:pos + SIZE_RR_MAX]),
                         key=operator.itemgetter(1))
            pos = (pos - SIZE_RR_MAX) + idx
        R_pos.append(pos)
        lab = -1
        if sym in MITBIH_CLASSES:
            for i, grp in enumerate(AAMI_CLASSES):
                if sym in grp:
                    lab = i
                    break
        if lab >= 0 and pos > WINL and pos < (len(mlii) - WINR):
            beats.append(mlii[pos - WINL:pos + WINR].copy())
            labels.append(lab)
            valid.append(1)
        else:
            valid.append(0)
    return beats, np.asarray(labels, np.int64), np.asarray(valid, np.int64), np.asarray(R_pos, np.int64)


def extract_104(mlii, anns):
    beats, labels, valid, R_pos = build_beats(mlii, anns)
    if len(beats) == 0:
        return (np.zeros((0, 104), np.float64), np.zeros(0, np.float64),
                np.zeros(0, np.int64))

    rr = compute_RR_intervals(R_pos)
    rr_pre = rr.pre_R[valid == 1]
    rr_post = rr.post_R[valid == 1]
    rr_local = rr.local_R[valid == 1]
    rr_global = rr.global_R[valid == 1]
    # 对齐保护（正常应严格等于 beat 数）
    n = min(len(beats), len(rr_pre))
    beats = beats[:n]; labels = labels[:n]
    rr_pre, rr_post, rr_local, rr_global = rr_pre[:n], rr_post[:n], rr_local[:n], rr_global[:n]

    f_rr = np.column_stack([rr_pre, rr_post, rr_local, rr_global])
    f_rr_norm = np.column_stack([
        rr_pre / max(np.average(rr_pre), 1e-12),
        rr_post / max(np.average(rr_post), 1e-12),
        rr_local / max(np.average(rr_local), 1e-12),
        rr_global / max(np.average(rr_global), 1e-12)])

    beats_arr = np.asarray(beats, dtype=np.float64)
    f_lbp = compute_Uniform_LBP_batch(beats_arr, 8)
    f_wav = np.vstack([compute_wavelet_descriptor(b, "db1", 3) for b in beats_arr])
    f_hos = compute_hos_descriptor_batch(beats_arr, 6,
                                         int(round((WINL + WINR) / 6)))
    f_morph = np.vstack([compute_my_own_descriptor(b, WINL, WINR)
                         for b in beats_arr])

    features = np.column_stack([f_rr, f_rr_norm, f_lbp, f_wav, f_hos, f_morph])
    return features, rr_pre, labels


if __name__ == "__main__":
    import os
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default=None, help="WFDB 记录前缀")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--txt", default=None)
    a = ap.parse_args()
    if a.record:
        mlii, _, anns = read_wfdb_record(a.record)
    else:
        mlii, _, anns = read_csv_record(a.csv, a.txt, int(os.path.basename(a.csv)[:3]))
    X, pre, y = extract_104(mlii, anns)
    print("features", X.shape, "pre_R", pre.shape, "labels", y.shape)
    print("first pre_R", pre[:5].tolist() if len(pre) else [])
    print("class counts", {i: int((y == i).sum()) for i in range(4)})
