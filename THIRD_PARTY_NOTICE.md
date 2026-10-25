# 第三方说明

## 1. 104 维特征管线

`feature_extract.py` 的 104 维特征定义和大部分数值逻辑，复刻自公开的 ECG 特征管线：

- Mondejar-Guerra, V., et al. *Heartbeat classification using disease-specific feature selection.*
  Computers in Biology and Medicine, 2019.
- 参考实现：`mondejar/ecg-classification`（GitHub），原仓库以 **GNU GPL-3.0** 发布。
- 本仓库的 `feature_extract.py` 是为独立推理重写的精简版：只保留 MLII、±90 窗、
  104 维特征所需函数，去掉了训练/数据库/绘图依赖；**不包含 Cardia 2.0 模型或训练框架源码**。
- 本仓库整体以 **GPL-3.0** 发布，以兼容上述来源。原仓库的版权归其作者所有；
  本仓库对特征管线的修改（批量 LBP/HOS、短记录 padding 等）同样以 GPL-3.0 提供。

## 2. 数据

- **MIT-BIH Arrhythmia Database**，PhysioNet：
  <https://physionet.org/content/mitdb/1.0.0/>
  许可：Open Data Commons Attribution License v1.0 (ODC-BY 1.0)。
  再分发数据时请保留署名；本仓库默认不把完整数据库提交进 git，
  用 `scripts/download_mitbih.py` 下载。

## 3. 自研部分

- Cardia 2.0 模型结构、训练流程、checkpoint 选择、104 维特征之上的序列建模与
  推理封装为本项目自研；特征管线的来源见第 1 节。
- README 中的基线数字是同一套 104 维特征、同一 DS1/DS2 划分下的**统一复现**，
  不是各论文的官方数字。

## 4. 引用

如果你在学术工作中使用本仓库，请同时引用：

- de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of
  heartbeats using ECG morphology and heartbeat interval features. IEEE TBME.
- Mondejar-Guerra, V., et al. (2019). Heartbeat classification using disease-specific
  feature selection. Computers in Biology and Medicine.
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  Circulation, 101(23), e215–e220.
