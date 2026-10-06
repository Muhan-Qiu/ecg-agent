# Usage Guide

> The interface is bilingual — switch languages from the top bar. This guide uses the English UI.
> 中文版见 [`使用说明.md`](使用说明.md)。

## 1. Running it

```bash
git clone https://github.com/Muhan-Qiu/ecg-agent.git
cd ecg-agent
pip install -r requirements.txt

streamlit run app.py          # web UI (main entry point)

# if you see "command not found: streamlit", use the equivalent form
python3 -m streamlit run app.py
```

Once the terminal prints `Local URL: http://localhost:8501`, the browser will open
automatically. Press `Control + C` in the terminal to shut it down.

**Command line** (when you don't want the web UI, or for a terminal demo):

```bash
python3 ecg_agent.py "Does record 208 need immediate referral?"
python3 ecg_agent.py          # no question → interactive mode, type q to quit
```

**Retraining the classifier** (only if you want to reproduce training; not needed day to day):

```bash
python3 train_classifier.py
```

## 2. Interface tour

Five controls across the top, left to right:

| Control | What it does |
|---|---|
| **中文 / English** | Switches language. Interface text, chart labels, the agent's answer language, and the tool descriptions sent to the model all switch together. Existing conversation history is not retranslated. |
| **Built-in record** | Pick one of the 44 MIT-BIH records. `100` is essentially normal; `208` and `233` have severe ventricular arrhythmia and make a good contrast. |
| **Upload** | Upload your own ECG CSV (see section 3). |
| **Sampling rate** | Only applies to uploads; irrelevant for built-in records. |
| **⚙ Model** | Choose an LLM provider and enter an API key (see section 4). |

A line below the bar shows the current model and endpoint, so you always know what you're talking to.

The body is split into two columns:

- **Left** is the conversation. Type in the input box at the bottom and press Enter. You will see `🔧 calling xxx(...)` appear in real time; expanding the collapsed panel shows the raw JSON each tool returned. **This panel is the single most useful thing to show in a presentation** — it demonstrates that the model is not inventing numbers, because every conclusion traces back to a specific tool call.
- **Right** is the charts: four metric cards (mean heart rate / rhythm / abnormal beats / duration), the waveform with detected R peaks marked, the per-beat classification donut, the raw measurement values, and the **Export report** button at the bottom.

**Export report** produces Markdown containing the record info, measurements, beat counts and the full conversation so far — ready to paste into a lab report.

## 3. Uploading your own ECG

Accepts a **single-column CSV** as exported by lab equipment (e.g. the M501 exported through ParamMonitor): plain numbers, no header, 12-bit raw ADC counts.

**How**: click the upload box and pick your file → enter **500** as the sampling rate → after a second or two a toast appears saying it was imported → the charts on the left switch to your data.

The uploaded file is converted to standard WFDB format and stored in `data/` (named `up_<hash>_<original name>`). From then on it behaves exactly like a built-in record — every tool can analyse it directly, and it appears in the record dropdown.

**Three things you need to know:**

1. **You must set the sampling rate by hand.** Exported CSVs carry no sampling-rate information. The data from this lab course is **500 Hz** — the evidence is that the T-wave peak appears about 190 ms after the R peak, which is the normal T-wave timing.
2. **The amplitude is not a real voltage.** The instrument provides no gain calibration, so the program percentile-scales the amplitude to roughly the mV range purely to keep filtering and display in a sensible range.
3. **Treat the classification with caution.** The classifier was trained on MIT-BIH's MLII lead. With a different lead (especially inverted polarity) waveform morphology changes. **Heart rate, rhythm regularity and HRV are unaffected** — they are time-domain quantities, independent of lead and amplitude.

## 4. Configuring the model

Click **⚙ Model** to open the panel:

1. **Provider** — pick from the dropdown; the base URL and model name are filled in automatically.
2. **Base URL / Model** — editable. Model names are reference values only; vendors rename them frequently.
3. **API Key** — your own. Greyed out for the local proxy and local Ollama, which need none.
4. Click **Apply and test** — fires a minimal request and tells you right away whether it works.

The default routes through a **local proxy** that reuses Claude Code's channel; it needs no API key but requires that proxy to be running.

**On key safety**: a key entered in the UI lives only in the current browser session's memory — never written to a file, never logged, never echoed back, and never embedded in the page source (Streamlit inputs are sent back to the local Python process rather than baked into HTML). **The real risk is hard-coding a key in a source file and committing it to GitHub** — once pushed, deleting the file does not help, because anyone can read your commit history. Once it's out there, the only remedy is to revoke the key at the provider and issue a new one.

If you want to configure from the environment instead, so that no credential ever appears in the code:

```bash
export ECG_LLM_FORMAT=openai
export ECG_LLM_BASE_URL=https://api.deepseek.com
export ECG_LLM_MODEL=deepseek-chat
export ECG_LLM_API_KEY=sk-xxxxxx
streamlit run app.py
```

## 5. Test checklist

Worth walking through before you publish:

- [ ] `streamlit run app.py` starts and the browser opens
- [ ] The Chinese interface renders with all text in Chinese
- [ ] Switching to English turns the whole UI English, including metric cards and chart labels
- [ ] Selecting record `100` shows **76 bpm / regular / 1.3% abnormal beats**
- [ ] Selecting `208` shows **98 bpm / irregular / 46.4% abnormal beats**
- [ ] Selecting `233` shows **102 bpm / irregular / 28.9% abnormal beats**
- [ ] Asking "Does this record need immediate referral?" returns an answer
- [ ] Expanding the tool-call panel above the answer shows arguments and JSON
- [ ] **Export report** downloads a file you can open in a text editor
- [ ] Uploading your own CSV with a sampling rate of 500 imports successfully
- [ ] Clicking ⚙ Model, selecting DeepSeek, entering a key and hitting "Apply and test" succeeds

## 6. Troubleshooting

**Connection errors in the chat area** → the local proxy requires that proxy app to be running. Start it, or switch to another provider in ⚙ Model and enter a key.

**A third-party provider fails "Apply and test"** → the test runs in two stages, so check which one fails:

- **Stage 1 (plain chat) fails** → network, base URL or API key problem. Check for stray spaces in the key and whether the account has credit.
- **Stage 1 passes, stage 2 (tool calling) fails** → the endpoint is reachable but **this model does not support function calling**. Switch to a model that does — for DeepSeek choose `deepseek-chat`, **not a reasoning model** (reasoning models generally do not support tool calling).

**Reaching overseas endpoints** → direct connections to the official Anthropic API typically fail from mainland China (the TLS handshake is blocked). Use a proxy, or switch to a domestic compatible endpoint.

**`command not found: streamlit`** → use `python3 -m streamlit run app.py`.

**`ModuleNotFoundError`** → you are using the wrong Python. Check that `which python3` points at the interpreter where you installed the requirements.

**Charts don't change after upload** → check that you entered a sampling rate and that the "imported" toast appeared.

**The agent keeps calling the same tool and gets stuck** → there is a 6-step cap and it will stop on its own. This usually means the model isn't capable enough (common with small local models); try a different provider.
