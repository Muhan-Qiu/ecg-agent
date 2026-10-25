# -*- coding: utf-8 -*-
"""复现 Cardia 2.0 的 DS2 指标并写出图/文档所需的 JSON。

用法（在仓库根目录）：
    python scripts/benchmark_ds2.py --v2 --detected

输出到 docs/benchmark/：
    v2_metrics.json / v2_confusion.json / v2_per_record.csv
    v2_detected_metrics.json / v2_detected_confusion.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, recall_score)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ecg_tools as T          # noqa: E402
import feature_extract as fe   # noqa: E402
import cardia_classifier as rc  # noqa: E402

DS2 = ("100 103 105 111 113 117 121 123 200 202 210 212 "
       "213 214 219 221 222 228 231 232 233 234").split()
LABELS = [0, 1, 2, 3]
NAMES = ["N", "SVEB", "VEB", "F"]
OUT = ROOT / "docs" / "benchmark"


def metrics(y, p):
    return {
        "n": int(len(y)),
        "acc": float(accuracy_score(y, p)),
        "kappa": float(cohen_kappa_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", labels=LABELS,
                                   zero_division=0)),
        "balanced_acc": float(recall_score(y, p, average="macro", labels=LABELS,
                                           zero_division=0)),
        "per_class_recall": {
            n: float(r) for n, r in zip(
                NAMES, recall_score(y, p, average=None, labels=LABELS,
                                    zero_division=0))
        },
        "confusion": confusion_matrix(y, p, labels=LABELS).tolist(),
    }


def write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def run_v2(detected=False):
    pp = rc.load_preprocess()
    model = rc.load_model(fmt="ts")
    all_true, all_pred, rows = [], [], []
    for rec in DS2:
        sig, _, anns, _ = rc._read_wfdb_digital(rec)
        if detected:
            peaks = rc._detect_peaks(sig, 360)
            anns = [(int(p), "N") for p in peaks]
        X, pre, labels = fe.extract_104(sig, anns)
        probs, _ = rc.predict_features(X, pre, pp, model, "ts")
        pred = probs.reshape(-1, probs.shape[-1]).argmax(axis=-1)

        if not detected:
            y = labels[:len(pred)]
        else:
            # 自动峰 -> 专家标注最近邻（50 ms = 18 samples）
            ann = wfdb.rdann(str(ROOT / "data" / rec), "atr")
            apos = np.asarray(ann.sample)
            asym = np.asarray(ann.symbol)
            keep = np.array([T.AAMI_MAP.get(s) in ("N", "S", "V", "F")
                             for s in asym])
            apos, asym = apos[keep], asym[keep]
            order = np.argsort(apos)
            apos, asym = apos[order], asym[order]
            _, _, valid, R_pos = fe.build_beats(sig, anns)
            valid_pos = R_pos[valid == 1][:len(pred)]
            idx = {"N": 0, "S": 1, "V": 2, "F": 3}
            y = []
            for p in valid_pos:
                j = np.searchsorted(apos, p)
                cand = [(abs(apos[k] - p), asym[k]) for k in (j - 1, j)
                        if 0 <= k < len(apos)]
                if cand:
                    dist, sym = min(cand, key=lambda x: x[0])
                    if dist <= 18:
                        y.append(idx[T.AAMI_MAP[sym]])
                        continue
                y.append(-1)
            y = np.asarray(y)
            m = y >= 0
            y, pred = y[m], pred[m]

        all_true.append(y)
        all_pred.append(pred)
        rows.append({"record": rec, "n": int(len(y)),
                     "acc": float(accuracy_score(y, pred))})

    y = np.concatenate(all_true)
    p = np.concatenate(all_pred)
    tag = "v2_detected" if detected else "v2"
    res = metrics(y, p)
    res["path"] = "detected_peaks" if detected else "wfdb_atr"
    write(f"{tag}_metrics.json", res)
    write(f"{tag}_confusion.json", res["confusion"])
    with open(OUT / f"{tag}_per_record.csv", "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["record", "n", "acc"])
        w.writeheader()
        w.writerows(rows)
    print(tag, {k: round(v, 4) if isinstance(v, float) else v
                for k, v in res.items() if k not in ("confusion",
                                                     "per_class_recall")})
    print("  recall", {k: round(v, 4) for k, v in
                       res["per_class_recall"].items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true", help="专家标注路径")
    ap.add_argument("--detected", action="store_true", help="自动检测 R 峰路径")
    a = ap.parse_args()
    if not (a.v2 or a.detected):
        a.v2 = a.detected = True
    if a.v2:
        run_v2(detected=False)
    if a.detected:
        run_v2(detected=True)
