# V1.0 Legacy (RandomForest)

V1.0 is the initial release. The complete source and original README are preserved at:

<https://github.com/Muhan-Qiu/ecg-agent/tree/v1.0.0>

This page explains what V1.0 was, its reported results, and how to keep using the V1.0
classifier in the current codebase.

## 1. What V1.0 was

- **LLM agent + 5 local tools**: `list_records`, `check_quality`, `measure`, `classify`,
  `lookup_guideline`.
- **Classifier**: RandomForest, 200 trees, 49-dimensional features, per-beat independent
  classification.
- **R peaks**: detected automatically with neurokit2; no expert annotations required.
- **Input**: MIT-BIH records, or single-column CSV from lab devices / smartwatches.
- **UI**: bilingual Streamlit interface with report export.

## 2. V1.0 evaluation (original protocol)

V1.0 was evaluated on DS2 with 48,751 automatically detected beats:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| N normal | 0.948 | 0.994 | 0.971 |
| S supraventricular ectopic | 0.686 | 0.039 | 0.074 |
| V ventricular ectopic | 0.935 | 0.904 | 0.919 |
| F fusion | 0.033 | 0.008 | 0.013 |
| **Overall accuracy** | | | **94.5%** |

V1.0 feature ablation (S recall / V recall / overall accuracy):

| Features | S recall | V recall | Overall accuracy |
|---|---:|---:|---:|
| RR intervals only | 0.070 | 0.755 | 0.918 |
| Morphology only | 0.008 | 0.816 | 0.927 |
| All features | 0.039 | 0.904 | 0.945 |

> **Do not compare these numbers directly with V2.0.** V1.0 uses 49-dimensional features
> and automatically detected R peaks, evaluated on 48,751 detected beats; V2.0 uses
> 104-dimensional features with annotated/detected R peaks, evaluated on 49,024
> annotation-aligned beats. The features, detection pipeline and evaluation beat sets
> differ. The public composite score of V2.0 is reported in the main README.

## 3. Using the V1.0 classifier in the current code

- CLI: `T.classify_beats(record, engine="fallback")`
- UI: select “RandomForest (fallback)”
- Retraining: `python train_classifier.py`
- Weights: `models/beat_classifier.pkl`
- Training script: `train_classifier.py`
- Features and tools: `ecg_tools.py`

## 4. Main limitations of V1.0

1. S recall is only 3.9%: supraventricular ectopic beats have nearly the same QRS
   morphology as normal beats and require P-wave information, which the 49-dimensional
   feature set does not contain.
2. F recall is only 0.8%: the F class has very few samples and is difficult to train.
3. The classifier was trained on MIT-BIH MLII; other leads, sampling rates or device gains
   cause performance loss.
4. Algorithmic pre-screen only; not a medical device and not for clinical decisions.
