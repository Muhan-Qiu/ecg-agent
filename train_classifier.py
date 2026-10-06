# -*- coding: utf-8 -*-
"""
训练心搏分类器

数据：MIT-BIH 心律失常数据库，采用 de Chazal (2004) 的标准划分
      DS1（22 条记录）训练，DS2（22 条记录）测试。
      这样划分保证训练集和测试集来自不同患者，才是真实场景下的性能。

标签：AAMI 五类。其中 Q 类全库只有 15 个心搏，样本太少，弃用，
      实际训练 N / S / V / F 四类。

运行：  python3 train_classifier.py
"""
import os
import json
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import ecg_tools as T

DS1 = ("101 106 108 109 112 114 115 116 118 119 122 124 "
       "201 203 205 207 208 209 215 220 223 230").split()
DS2 = ("100 103 105 111 113 117 121 123 200 202 210 212 "
       "213 214 219 221 222 228 231 232 233 234").split()

MODEL_DIR = os.path.join(T.HERE, "models")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print(f"训练集 {len(DS1)} 条记录，测试集 {len(DS2)} 条记录")
    t0 = time.time()
    Xtr, ytr = T.collect_dataset(DS1, verbose=True)
    print(f"训练集特征矩阵 {Xtr.shape}，耗时 {time.time() - t0:.0f}s")

    t1 = time.time()
    Xte, yte = T.collect_dataset(DS2, verbose=True)
    print(f"测试集特征矩阵 {Xte.shape}，耗时 {time.time() - t1:.0f}s")

    print("\n训练随机森林……")
    model = RandomForestClassifier(
        n_estimators=200,           # 树的数量：再多收益很小，200 棵够用
        min_samples_leaf=2,
        class_weight="balanced_subsample",   # 四类样本量差几十倍，需要加权
        n_jobs=-1,
        random_state=0,
    )
    model.fit(Xtr, ytr)
    print(f"训练完成，耗时 {time.time() - t1:.0f}s")

    T.evaluate(model, Xte, yte, title="DS2 测试集（不同患者）")

    with open(os.path.join(MODEL_DIR, "beat_classifier.pkl"), "wb") as f:
        pickle.dump(model, f)
    with open(os.path.join(MODEL_DIR, "split.json"), "w") as f:
        json.dump({"train": DS1, "test": DS2}, f, indent=2)
    print(f"\n模型已保存到 {MODEL_DIR}/beat_classifier.pkl")


if __name__ == "__main__":
    main()
