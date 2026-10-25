# ECG Agent · v2.0

**Cardia 2.0: ECG-specific beat classification with traceable tool calls**

**English** (this file) ｜ **中文**: [`README.md`](README.md)

---

## 1. Overview

ECG Agent is an open-source project that combines a large language model with local ECG
analysis tools. The LLM selects tools and explains results; all signal processing,
R-peak detection, heart-rate/HRV measurement, beat classification and critical-value
lookup run in local tools. Every reported number comes from a tool return value and is
traceable.

Version **v2.0** uses the self-developed **Cardia 2.0** beat classifier. Version **v1.0**
used a RandomForest classifier.

## 2. Version history

| Version | Classifier | Main changes |
|---|---|---|
| v1.0 | RandomForest (49-dim, per-beat) | Initial release; 5 local tools; bilingual UI; single-column CSV upload |
| v2.0 | Cardia 2.0 (104-dim, 64-beat sequence) | Self-developed model; composite score 74.00/100; model card; experimental upload path |

- V1.0 source and original README: <https://github.com/Muhan-Qiu/ecg-agent/tree/v1.0.0>
- V1.0 summary: [`docs/v1-legacy.en.md`](docs/v1-legacy.en.md)

## 3. v2.0 update and Cardia 2.0

**v2.0 update**: the beat classifier has been replaced by the self-developed
**Cardia 2.0**. On the test set (de Chazal DS2), its composite score is
**74.00 (out of 100)**.

Cardia 2.0 is the ECG-specific beat classifier developed in this project. It takes
104-dimensional beat features over a 64-beat sequence and outputs four-class
probabilities: N / SVEB / VEB / F.

### 3.1 Design rationale (overview)

Existing multi-layer bidirectional recurrent networks (e.g., multi-layer BiLSTM)
propagate both forward and backward states between layers. As depth increases, each layer
mixes information from both directions, making the inference-time boundary of "what is
visible at the current beat" unclear. For beat-by-beat ECG analysis, this bidirectional
mixing complicates online inference and interpretation. In addition, deep recurrent
stacks without cross-layer shortcuts tend to attenuate information and gradients across
layers; simply adding depth increases the parameter count and worsens overfitting on
small ECG datasets.

### 3.2 Design goals (overview)

Cardia 2.0 is designed around three goals:

1. **Causal inference-time information flow**: do not rely on future beats, so the causal
   boundary of beat-by-beat interpretation remains clear.
2. **Stable deep sequence modeling**: provide more direct cross-layer information paths
   for multi-layer state updates, avoiding the attenuation caused by simply stacking
   layers.
3. **Controlled model size and seed variance**: improve the stability of multi-layer
   sequence modeling and minority-class discrimination without significantly increasing
   the parameter count.

> This section describes the motivation and goals only. The specific inter-layer
> connections, gating forms, state-update rules, training strategy, checkpoint-selection
> criteria and feature/model co-design are not disclosed in public materials. Public
> weights are released as ONNX / TorchScript; the preprocessing contract is in
> `models/cardia/preprocess.json`.

## 4. Composite score (out of 100)

The table compares Cardia 2.0 with 18 deep-neural baselines on the test set. All models
use the same 104-dimensional features and the de Chazal DS1/DS2 split; baseline values
are 5-seed means.

| Model | n | Composite score (out of 100) |
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

### Scoring method

**Composite score = 0.40·Ijk + 0.30·Acc + 0.20·Macro-F1 + 0.10·BalAcc**, out of 100.

Ijk uses the corrected AAMI evaluator convention (V-class false positives use
`conf_mat[3][2]`). Acc is overall accuracy, Macro-F1 is the arithmetic mean of the four
class F1 scores, and BalAcc is the mean of the four class recalls.

> The released checkpoint is a single model; the same configuration's 5-seed mean is
> **70.02 (out of 100)**. Data:
> [`docs/benchmark/score_table.csv`](docs/benchmark/score_table.csv).

![Score comparison](docs/assets/fig_score_bar.png)

> All baselines are sorted by composite score; Cardia 2.0 is shown in red. Baselines are
> 5-seed means; the released model is a single checkpoint.

## 5. Evaluation protocol

| Item | Setting |
|---|---|
| Data | MIT-BIH Arrhythmia Database |
| Split | de Chazal (2004) inter-patient: DS1 22 records train, DS2 22 records test |
| Classes | AAMI four classes: N / SVEB / VEB / F |
| Input | MLII digital signal, 360 Hz; two median filters; R-peak ±20 local maximum; ±90 window |
| Features | 104-dim: RR 8 + u-LBP 59 + Wavelet 23 + HOS 10 + myMorph 4 |
| Chunking | 64-beat non-overlapping chunks within each record |
| Selection | DS1 3-fold validation, `Kappa + 0.25·SVEB recall` (3-epoch smoothing) |
| Test set | Evaluated once; not used for tuning or model selection |
| Baselines | Same 104-dim features and split; 5-seed mean |

## 6. System architecture

![System architecture](docs/assets/fig0_architecture.png)

The system has two layers:

- **Data/model layer**: raw ECG is preprocessed into 104-dimensional features, split into
  64-beat non-overlapping chunks, and passed to Cardia 2.0 to produce per-beat four-class
  probabilities.
- **Agent layer**: local tools package classification results, measurements, quality
  checks and guideline entries as JSON; the LLM only selects tools and explains results.
- **Traceability**: every number comes from a tool return value; tool inputs and raw JSON
  can be expanded in the UI.

### Tools

| Tool | Purpose |
|---|---|
| `list_records` | List available records |
| `check_quality` | Lead detachment, artifact, high-frequency noise |
| `measure` | Heart rate, rhythm regularity, HRV |
| `classify` | Per-beat four-class classification; `engine=auto/v2/fallback` |
| `lookup_guideline` | Critical-value guideline entries |
| `model_info` | Model card: parameter count, composite score, scope |

## 7. Quick start

```bash
git clone https://github.com/Muhan-Qiu/ecg-agent.git
cd ecg-agent
pip install -r requirements.txt

# Download MIT-BIH (default: data/; data is not committed)
python scripts/download_mitbih.py

streamlit run app.py
```

Open `http://localhost:8501` in a browser.

**Command line**

```bash
python ecg_agent.py "Does record 208 need immediate referral?"
python -c "import ecg_tools as T; print(T.classify_beats('208', engine='v2'))"
```

**Reproduce the benchmark and figures**

```bash
python scripts/benchmark_ds2.py --v2
python scripts/make_plots.py
```

**LLM configuration**: open ⚙ Model in the top-right corner, choose a provider, enter an
API key, and click “Apply and test”. Ten provider presets are included (DeepSeek / OpenAI /
Anthropic / Moonshot / Zhipu / Qwen / MiniMax / local Ollama / custom). API keys are held
only in session memory; they are never written to disk, logged or echoed.

## 8. Model size and runtime

| Item | Cardia 2.0 |
|---|---:|
| Parameters | 809,332 |
| Weights | ONNX 5.6 MB / TorchScript 3.6 MB |
| Input | 104 features/beat × 64 beats |
| Context | 64-beat sequence |
| CPU inference | About 2–4 s per 30-minute record (TorchScript, warmed up) |

> The ONNX graph has about 24k nodes; first session creation can take about one minute on
> some Windows machines. TorchScript loads in about two seconds. The app prefers
> TorchScript and falls back to ONNX when PyTorch is unavailable.

## 9. Upload and formats

Single-column CSV with a metadata header is supported (validated with a Samsung Galaxy
Watch export):

```text
名称,示例用户
采样率,500.000 Hz
导联,Lead I
单位,mV

-0.108716
-0.10939
...
```

Processing steps:

1. Skip non-numeric header lines; read sampling rate, lead and unit metadata;
2. Resample to 360 Hz;
3. Detect R peaks automatically;
4. Calibrate amplitude to the MIT-BIH digital scale;
5. Optional manual polarity inversion;
6. Run Cardia 2.0 and mark the result `experimental=true`.

> Cardia 2.0 was trained on MLII at 360 Hz; other leads, sampling rates and device gains
> are out of distribution. A 30-second watch recording usually has fewer than 64 beats;
> the system zero-pads to 64 beats and marks `padded=true`. Multi-format ingestion (WFDB,
> EDF, SCP-ECG, HL7 aECG, DICOM, etc.) is described in
> [`docs/roadmap.md`](docs/roadmap.md).

## 10. Scope and limitations

1. Research and education only; not a medical device and not for clinical diagnosis,
   triage or treatment decisions.
2. Evaluated only on the single MIT-BIH dataset and the single de Chazal DS1/DS2 split;
   no external validation.
3. Trained on MLII at 360 Hz; changing lead, sampling rate or device gain causes domain
   shift.
4. The composite score is an overall metric and does not replace per-class clinical
   validation; full per-class metrics and confusion matrices are kept in the internal
   evaluation record.

## 11. Data and license

- **MIT-BIH Arrhythmia Database** (44 records, about 100k beats)
  PhysioNet: <https://physionet.org/content/mitdb/1.0.0/>
  License: Open Data Commons Attribution License v1.0 (ODC-BY 1.0).
- **Guideline**: [`guideline.md`](guideline.md), excerpted from the 2017 Chinese expert
  consensus on ECG critical values.
- **Code**: GPL-3.0; see [`LICENSE`](LICENSE). The provenance of the 104-dimensional
  feature pipeline and modifications are documented in
  [`THIRD_PARTY_NOTICE.md`](THIRD_PARTY_NOTICE.md).

## 12. Citation

If you use this repository in academic work, please cite:

- de Chazal, P., O'Dwyer, M., & Reilly, R. B. (2004). Automatic classification of
  heartbeats using ECG morphology and heartbeat interval features. IEEE TBME.
- Mondejar-Guerra, V., et al. (2019). Heartbeat classification using disease-specific
  feature selection. Computers in Biology and Medicine.
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
  Circulation, 101(23), e215–e220.
