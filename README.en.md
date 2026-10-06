# ECG Agent

**心电图智能体 — an LLM agent that reads electrocardiograms, where every number is traced to a tool call**

> A medical agent where the LLM plans and explains while local tools do the computing: the model decides which tool to call and translates results into a clinical interpretation, and **every number comes from a tool return value — the model is not allowed to produce any of its own.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Data: ODC-BY 1.0](https://img.shields.io/badge/data-ODC--BY%201.0-lightgrey.svg)](https://physionet.org/content/mitdb/1.0.0/)

**中文版**：[`README.md`](README.md) ｜ **English** (this file)

- 🌐 **Bilingual UI** — interface text, chart labels, the agent's answers and the tool descriptions all switch language together
- 📖 **User guide**：[`USAGE.en.md`](USAGE.en.md)
- 🫀 **Bring your own ECG** — accepts single-column CSVs exported by lab equipment
- 🔍 **Fully transparent** — the arguments and raw JSON of every tool call are visible

---

## Screenshots

<!--
Take two screenshots and drop them in assets/, then uncomment:

![Chinese UI](assets/ui-zh.png)
![Tool calls in the conversation](assets/tool-calls.png)
-->

Waveform and R-peak detection, rendered by the program:

![Waveform and R-peak detection](assets/ecg-100.png)

---

## Why not just hand the ECG to a multimodal LLM and ask?

You can paste an ECG into a multimodal model and ask "what rhythm is this?" and you will get a fluent answer. But you have no reason to trust it — it might call the QRS a T wave, or invent a heart rate out of thin air, and you would have no way to tell.

**In medicine, "looks right" and "verifiable" are two different things.**

This project separates the responsibilities:

| Who | Does what |
|---|---|
| **The LLM** | Decides which tool to call next and with what arguments; turns tool return values into a structured clinical interpretation; answers follow-up questions |
| **Local tools** | All medical computation: filtering, R-peak detection, heart rate and HRV, beat classification, critical-value lookup |

Three constraints are hard-coded into the system prompt: **every number must come from a tool return value**, any abnormality requires quoting the guideline entry verbatim, and the answer must always end by stating that a physician must review it.

When a tool fails, the model does not fabricate — it reports the value as unavailable. The upshot: **every claim can be traced back to a specific tool call and its JSON return value.**

## Architecture

```
                        ┌──────────────────────────┐
   clinician  ────────► │  LLM (any OpenAI- or     │
   asks                │  Anthropic-compatible    │
                        │  endpoint)               │
                        └───────────┬──────────────┘
                                    │ tool_call
                        ┌───────────▼──────────────┐
                        │  Main loop (~50 lines)   │
                        │  run tool → append result│
                        └───────────┬──────────────┘
                                    │
     ┌──────────────┬───────────────┼───────────────┬──────────────┐
     ▼              ▼               ▼               ▼              ▼
 list_records  check_quality    measure        classify     lookup_guideline
                                HR / rhythm     per-beat      critical-value
                                HRV             random forest guideline entries
```

The **five tools** are just five ordinary Python functions. Swap the model or the vendor and not one line of the tool layer changes.

## Quick start

```bash
git clone https://github.com/Muhan-Qiu/ecg-agent.git
cd ecg-agent
pip install -r requirements.txt

streamlit run app.py                          # web UI
python3 ecg_agent.py "Does record 208 need immediate referral?"   # CLI
```

Then open `http://localhost:8501`.

**Configuring a model**: click **⚙ Model** in the top right, pick a provider, paste your API key, and hit "Apply and test" — it fires a minimal request and tells you immediately whether it works. Ten provider presets are built in (DeepSeek / OpenAI / Anthropic / Moonshot / Zhipu / Qwen / MiniMax / local Ollama / custom). The default routes through a local proxy and needs no key at all.

> Your API key is held only in session memory — never written to a file, never logged, never echoed back.

## Performance

The classifier is trained on the **MIT-BIH Arrhythmia Database** using the standard **inter-patient split** of de Chazal (2004) — 22 records for training, 22 for testing. Training and test sets come from **different patients**, so these numbers reflect real-world performance rather than the inflated figures a random split would give.

**DS2 test set, 48,751 beats:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| N (normal) | 0.948 | 0.994 | **0.971** |
| S (supraventricular ectopic) | 0.686 | 0.039 | 0.074 |
| V (ventricular ectopic) | 0.935 | 0.904 | **0.919** |
| F (fusion) | 0.033 | 0.008 | 0.013 |
| **Overall accuracy** | | | **94.5%** |

## Known limitations (this section matters more than the one above)

**Recall for the S class is only 3.9%, and this is an inherent shortcoming of the method, not something tuning will fix.**

I ran an ablation to confirm that:

| Features used | S recall | V recall | Overall accuracy |
|---|---|---|---|
| RR-interval features only | 0.070 | 0.755 | 0.918 |
| Waveform morphology only | 0.008 | 0.816 | 0.927 |
| All features | 0.039 | 0.904 | 0.945 |

The reason: **a supraventricular ectopic beat has almost the same QRS morphology as a normal beat.** What distinguishes it is the timing of the P wave (a premature ectopic P′, or an absent P). This feature set contains RR intervals and QRS morphology and **no P-wave information at all**. Published work reports similarly poor recall for this class.

The F class (fusion beats) fails for the same reason, compounded by having only 803 examples in the entire database — nowhere near enough to train on.

**Other boundaries:**

- The output is an **algorithmic pre-screen** and must not be used clinically without physician review
- When uploading your own data, if the lead differs from MIT-BIH's MLII (especially with inverted polarity), classification degrades noticeably. **Heart rate, rhythm regularity and HRV are unaffected**, since those are time-domain quantities
- Recordings made during exercise can be misread as "irregular rhythm" because motion artifact causes missed R peaks — that is a signal-quality problem, not a real arrhythmia
- The agent stops after 6 steps and says so

## Data attribution

- **MIT-BIH Arrhythmia Database** (44 records, 103,724 beats)
  PhysioNet: <https://physionet.org/content/mitdb/1.0.0/>
  License: Open Data Commons Attribution License v1.0 (ODC-BY 1.0)
- **Guideline**: [`guideline.md`](guideline.md) excerpts the *2017 Chinese Expert Consensus on
  ECG Critical Values* (Chinese Society of Pacing and Electrophysiology,
  *Journal of Clinical Electrocardiology* 2017;26(6))

The UI accepts single-column CSVs exported from lab equipment (12-bit raw ADC counts). **Uploaded files are excluded by `.gitignore` and never enter the repository.**

## Repository layout

```
ecg_tools.py         Toolset: loading, CSV import, quality check, filtering,
                     R-peak detection, measurement, classification, guideline lookup
ecg_agent.py         The agent itself: tool table + tool descriptions + main loop
                     + multi-vendor LLM support
app.py               Streamlit UI (bilingual)
train_classifier.py  Trains the beat classifier
guideline.md         Critical-value entries
models/              Trained classifier
data/                MIT-BIH records
```

## License

Code released under the [MIT License](LICENSE).
Bundled data remains under its original license (MIT-BIH is ODC-BY 1.0);
please preserve attribution when redistributing.

---

<sub>Course project for *Principles and Applications of Medical AI*, Biomedical Engineering, Chongqing University</sub>
