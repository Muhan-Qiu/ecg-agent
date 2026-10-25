# 测试集评测：综合评分与基线对比

本文件记录 Cardia 2.0 的综合评分、全部深层神经网络基线、评测协议与复现方式。
综合评分满分 100；逐类指标与混淆矩阵保存在内部评测记录中。

---

## 1. 评测协议

| 项 | 设置 |
|---|---|
| 数据 | MIT-BIH Arrhythmia Database |
| 划分 | de Chazal (2004) inter-patient：DS1 22 条训练，DS2 22 条测试 |
| 类别 | AAMI 四类：N / SVEB / VEB / F |
| 输入 | MLII 数字量，360 Hz；两次中值滤波；R 峰 ±20 局部最大；±90 窗 |
| 特征 | 104 维：RR 8 + u-LBP 59 + Wavelet 23 + HOS 10 + myMorph 4 |
| 切块 | 每条记录内 64 beat 非重叠 chunk |
| 选模 | DS1 3-fold 验证集，`Kappa + 0.25·SVEB recall`（3-epoch 平滑） |
| 测试集 | 只评一次，未用于调参或选模 |
| 基线 | 同一套 104 维特征、同一划分；5-seed 均值 |

---

## 2. 综合评分方式

**综合评分 = 0.40·Ijk + 0.30·Acc + 0.20·Macro-F1 + 0.10·BalAcc**，满分 100。

- **Ijk**：AAMI 评测器的 inter-class agreement 指标。原评估器将 V 类 FP 写为
  `conf_mat[i][3]`，正确应为 `conf_mat[3][2]`；本仓库使用修正后的 `Ijk_corr`。
- **Acc**：总体准确率。
- **Macro-F1**：四类 F1 的算术平均（`zero_division=0`）。
- **BalAcc**：四类召回的平均。

---

## 3. 综合评分（满分 100）

| 模型 | n | 综合评分（满分 100） |
|---|---:|---:|
| **Cardia 2.0 (released)** | 1 | **74.00** |
| T-LSTM | 5 | 70.69 ± 1.59 |
| Highway LSTM L4 | 5 | 70.38 ± 1.75 |
| Highway LSTM L2 | 5 | 69.89 ± 1.97 |
| BiLSTM L4 | 5 | 68.71 ± 0.89 |
| Highway BiLSTM L2 | 5 | 68.36 ± 2.85 |
| BiLSTM L2 | 5 | 67.49 ± 1.75 |
| BiLSTM dense L2 | 5 | 67.34 ± 2.05 |
| Highway BiLSTM L4 | 5 | 67.03 ± 1.14 |
| BiLSTM dense L4 | 5 | 66.93 ± 1.23 |
| BiLSTM resgate L4 | 5 | 66.74 ± 1.27 |
| BiLSTM resgate L2 | 5 | 65.55 ± 2.42 |
| BiLSTM (1 layer) | 5 | 65.37 ± 1.01 |
| InceptionTime | 5 | 64.31 ± 2.72 |
| BiLSTM residual L2 | 5 | 64.28 ± 0.83 |
| BiLSTM LN L2 | 5 | 64.24 ± 1.59 |
| ResNet1D | 5 | 64.14 ± 2.49 |
| BiLSTM residual L4 | 5 | 63.79 ± 1.15 |
| BiLSTM LN L4 | 5 | 63.72 ± 1.34 |

> 发布行为单 checkpoint；同配置 5-seed 均值为 **70.02（满分 100）**。
> 表中为全部 18 个深层神经网络基线。

![Score comparison](assets/fig_score_bar.png)

> 全部基线按综合评分排序；Cardia 2.0 为红色。

---

## 4. 复现

```bash
python scripts/download_mitbih.py
python scripts/benchmark_ds2.py --v2
python scripts/make_plots.py
```

`scripts/benchmark_ds2.py` 复现 Cardia 2.0 发布 checkpoint 的测试集结果；
基线评分来自同一 104 维特征、同一 DS1/DS2 划分的 5-seed 评测。

---

## 5. 评测限制

1. 发布权重为单 checkpoint；同配置 5-seed 均值为 70.02（满分 100）。
2. 基线评分来自同一套 104 维特征与同一 DS1/DS2 划分的统一复现，不是各论文官方数字。
3. 模型仅在 MIT-BIH 单一数据集、单一患者级划分上评测，尚无外部验证。
4. 综合评分为整体指标，不替代逐类临床验证；完整逐类指标与混淆矩阵保存在内部评测记录中。
5. 本系统为科研与教学用途，不是医疗器械，不能用于临床决策。
