# Model Card — Cardia 2.0

## 基本信息

| 项 | 内容 |
|---|---|
| 名称 | **Cardia 2.0**（心电专用模型） |
| 版本 | 2.0 |
| 类型 | 逐搏四类心电分类器（ECG-specific sequence model） |
| 结构 | 自研多层序列模型（具体层间连接与门控不公开） |
| 参数量 | 809,332 |
| 权重 | `models/cardia/cardia_k4_best.onnx`（5.6 MB）<br>`models/cardia/cardia_k4_best_ts.pt`（3.6 MB） |
| 输入 | 104 维/搏特征，每条记录内 64 beat 非重叠 chunk；`x [B,64,104]`，`dt [B,64]` |
| 输出 | 每搏四类 logits / softmax：`N / SVEB / VEB / F` |
| 任务 | AAMI 四类心搏分类 |
| 许可 | GPL-3.0（与仓库一致；特征管线来源见 `THIRD_PARTY_NOTICE.md`） |

## 训练数据

- **MIT-BIH Arrhythmia Database**，de Chazal (2004) 标准 inter-patient 划分：
  - DS1：22 条记录，训练；
  - DS2：22 条记录，测试；DS2 只评一次，未用于调参/选模。
- 类别：AAMI `N / SVEB / VEB / F`；`Q` 等样本过少的类别丢弃。
- 输入管线：MLII 数字量、360 Hz、两次中值滤波、R 峰 ±20 局部最大、±90 窗、104 维特征。
- 每条记录内 64 beat 非重叠 chunk；训练/评估不跨记录拼接。

## 评测

发布 checkpoint 由 DS1 3-fold 验证集按
`Kappa + 0.25·SVEB recall`（3-epoch 平滑）选择；DS2 只评一次。

**综合评分（满分 100）= 74.00**

综合评分 = 0.40·Ijk + 0.30·Acc + 0.20·Macro-F1 + 0.10·BalAcc，满分 100。

> 发布权重为单 checkpoint；同配置 5-seed 均值为 **70.02（满分 100）**。
> 完整逐类指标与混淆矩阵保存在内部评测记录中。

## 预期用途

- 教学、科研、算法原型验证；
- 作为“LLM + 本地工具”智能体中的逐搏分类组件；
- 在 MIT-BIH 类似的 MLII、360 Hz 数据上做离线评测。

## 不适用 / 超出范围

- **不是医疗器械**，不能用于临床诊断、分诊或治疗决策；
- 不能替代医生复核；
- 不适用于非 MLII 导联、非 360 Hz、未校准幅度、未标注/未检测 R 峰的数据（上传路径仅实验性）；
- 不适用于起搏心律、宽 QRS、严重伪差等训练分布外的记录。

## 已知偏差与风险

- **类别不平衡**：N 类占绝大多数，少数类样本有限；综合评分为整体指标，不替代逐类验证。
- **单数据集**：仅 MIT-BIH；不同人群、设备、导联、采样率会带来域偏移。
- **单划分**：仅 de Chazal DS1/DS2；没有外部验证或患者级交叉验证的第二张表。
- **checkpoint 选择**：单 checkpoint 的综合评分高于 5-seed 均值；存在选择方差。
- **特征管线来源**：104 维特征复刻自 GPL-3.0 公开工作；自研部分是 Cardia 2.0
  模型/训练/选模。
- **LLM 依赖**：对话能力来自第三方 LLM；模型性能数字必须通过 `model_info` 工具获取。

## 复现

```bash
python scripts/download_mitbih.py
python scripts/benchmark_ds2.py --v2
python scripts/make_plots.py
```

## 引用

见 `THIRD_PARTY_NOTICE.md` 与 README 的引用部分。
