# Changelog

## [2.0.0] — 2026-10-24

### Added

- **Cardia 2.0** 自研心电逐搏分类器（0.81M 参数，104 维特征，64-beat 序列）。
- ONNX（5.6 MB）与 TorchScript（3.6 MB）发布权重，`models/cardia/`。
- `cardia_classifier.py`：权重加载、104 维特征、每记录 64-beat 切块、短记录 padding。
- `classify(record, engine=auto/v2/rf)`：默认优先 Cardia 2.0，失败回退随机森林。
- `model_info` 工具：模型卡、参数量、DS2 指标、限制；性能数字只能来自工具。
- Streamlit 双模型切换与“对比”视图；导出报告包含模型/引擎/标注来源/警告。
- 上传路径实验性支持 Cardia 2.0：跳过厂商元数据头、读采样率、重采样 360 Hz、
  自动检测 R 峰、幅度校准、可选手动反转极性。
- `scripts/benchmark_ds2.py`、`scripts/make_plots.py`、`scripts/download_mitbih.py`。
- `docs/benchmark.md`、`docs/v1-vs-v2.md`、`docs/roadmap.md`、`MODEL_CARD.md`。
- `docs/assets/fig0–fig5` + `fig_hero_update`（PNG + SVG）。

### Changed

- 104 维特征提取的 LBP/HOS 改为批量向量化：单条 30 分钟记录从 ~8 s 降到 ~1.2 s，
  与原始实现逐元素一致（max diff 3.6e-15），预测 argmax 完全一致。
- 模型加载增加进程内缓存；有 torch 时默认 TorchScript（加载 ~2 s），
  否则回退 ONNX（首次建会话在部分机器上较慢）。
- 许可证从 MIT 改为 **GPL-3.0**，以兼容 104 维特征管线的 GPL-3.0 来源。
- README 重写为 v2.0 综合评分与评测说明。

### Performance (DS2, inter-patient)

- 发布 checkpoint：**综合评分 74.00（满分 100）**。
- 同配置 5-seed 均值：**综合评分 70.02（满分 100）**。

### Known issues

- 发布权重是单 checkpoint；5-seed 均值 70.02 低于单权重 74.00（满分 100）。
- 综合评分为整体指标，不替代逐类临床验证。
- 上传路径为实验性；非 MLII、非 360 Hz、短记录都有域偏移。
- 单数据集、单划分、无外部验证；不是医疗器械。

## [1.0.0] — 2026-06

- 初始版本：LLM agent + 5 个本地工具 + RandomForest 逐搏分类器。
- 中英双语 Streamlit 界面、导出报告、多厂商 LLM 接入。
