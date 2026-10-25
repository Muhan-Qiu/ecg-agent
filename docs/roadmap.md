# v2.1 路线图：多格式 ECG 自动接入（草案，待讨论）

> v2.0 只做“带元数据头的单列 CSV”实验性上传；本文件是 v2.1 的设计草案，
> **不是已实现功能**。厂商格式需要逐家核实官方导出说明/样本文件后再写进 README。

## 1. 目标

不管输入是医院 12 导联、手表 30 秒单导联、实验设备 CSV，还是 WFDB/EDF/SCP-ECG，
都先统一成一个内部契约，再进入现有工具链：

```python
@dataclass
class EcgRecord:
    signal: np.ndarray          # [n_samples, n_leads]，统一 mV
    fs: float                   # Hz
    leads: list[str]            # I/II/III/aVR/aVL/aVF/V1..V6/MLII/unknown
    annotations: list | None    # R 峰、标签（可选）
    metadata: dict              # 厂商、型号、导出时间、原始单位、增益
    warnings: list[str]         # 单位可疑、导联缺失、极性反转、采样率推断
```

## 2. 分层

```text
ingest/
├── base.py          # EcgRecord / Annotation / Adapter 协议
├── detect.py        # 扩展名 + magic bytes + 内容嗅探
├── adapters/
│   ├── wfdb.py
│   ├── edf.py
│   ├── scp_ecg.py          # EN 1064，医院 12 导联常用
│   ├── hl7_aecg.py         # XML
│   ├── dicom_ecg.py
│   ├── csv_generic.py      # 列名/分隔符/单位可配置
│   ├── apple_health.py     # Health export XML + electrocardiograms/*.csv
│   ├── fit.py              # Garmin/Polar 等
│   ├── json_generic.py
│   └── pdf_best_effort.py  # 最后手段，必须人工确认
├── normalize.py     # 单位、导联映射、极性、重采样、时间轴
└── validate.py      # 时长、采样率、NaN、平线、饱和、导联脱落
```

每个 adapter 两个方法：`can_handle(file) -> bool`、`load(file) -> EcgRecord`。
新厂商只加一个 adapter，不动主流程。

## 3. 自动处理流程（带人工兜底）

```text
上传 → 格式嗅探 → 解析元数据(fs/导联/单位/时长)
     → 置信度高：自动归一化 → 质量检查 → 分析
     → 置信度低：弹出“解析结果”面板，允许改采样率/导联顺序/单位/极性
     → 存 normalized record + 溯源信息 → 后续所有工具直接可用
```

原则：**能自动就自动，不确定就让用户确认一次**。医学数据不要猜采样率和极性。

## 4. 优先级

| 优先级 | 格式 | 理由 |
|---|---|---|
| P0 | WFDB、通用 CSV、EDF/EDF+ | 现有数据、实验设备、公开数据集 |
| P1 | SCP-ECG、HL7 aECG、DICOM ECG、PDF | 医院 12 导联常见导出 |
| P2 | Apple Health、Samsung Health、华为、小米/Zepp、Garmin/FIT | 手表/消费级设备 |
| P3 | 厂商专有格式（AcqKnowledge/LabChart 等） | 实验设备，按需 |

## 5. 厂商候选清单（待逐家核实）

| 类别 | 厂商/设备 | 常见导出 | 备注 |
|---|---|---|---|
| 手表 | Apple Watch | Health export XML + `electrocardiograms/*.csv`、PDF | ~512 Hz、30 s、单导联 |
| 手表 | Samsung Galaxy Watch | Samsung Health CSV/JSON、PDF | ~500 Hz、单导联；本机样本已验证（不上传个人数据） |
| 手表 | 华为 Watch | 运动健康 CSV/JSON、PDF | 国行导出路径不同 |
| 手表 | 小米/Zepp、Withings、Garmin | CSV/JSON/PDF/FIT | 多为单导联 30–60 s |
| 单导联 | AliveCor Kardia、Omron | CSV/PDF | PDF 需解析 |
| 医院 12 导联 | 迈瑞、理邦、科曼、鱼跃、乐普 | SCP-ECG/PDF/XML/DICOM | 500/1000 Hz，12 导联 |
| 通用标准 | WFDB、EDF+、SCP-ECG、HL7 aECG、DICOM | — | 优先支持公开标准 |

> 这张表是候选，不是最终结论。正式实现前需要：
> 1. 找到每家官方导出说明或真实样本；
> 2. 确认单位（mV/ADC）、采样率、导联顺序、极性；
> 3. 至少用 1 条真实记录跑通 `EcgRecord`；
> 4. 在 README 里标注“已验证/未验证”。

## 6. 非目标

- 不做“任意 PDF 都能完美解析”的承诺；PDF 只做 best-effort，必须人工确认。
- 不替代医院 PACS/HIS 的正式接口。
- 不把未验证的厂商格式写成“已支持”。
