<div align="center">

# 📡 ML-Based Adaptive Modulation Scheme

**Automatically select the optimal modulation scheme using Machine Learning**

> *A machine learning pipeline that intelligently selects BPSK, QPSK, 16-QAM, or 64-QAM based on real-time channel quality — maximising spectral efficiency without sacrificing link reliability.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Concept](#-concept)
- [Features](#-features)
- [Modulation Decision Logic](#-modulation-decision-logic)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Notebook Sections](#-notebook-sections)
- [Results](#-results)
- [Models Used](#-models-used)
- [Dependencies](#-dependencies)
- [Author](#-author)

---

## 🔭 Overview

In wireless communication systems, fixed modulation schemes waste spectral resources during favourable channel conditions and fail under poor ones. This project implements a **Machine Learning–based Adaptive Modulation System** that:

- Takes **SNR** (Signal-to-Noise Ratio) and **BER** (Bit Error Rate) as inputs
- Automatically outputs the **optimal modulation scheme** in real time
- Achieves up to **6× spectral efficiency gain** over fixed BPSK

---

## 💡 Concept

```
Channel Quality Inputs          ML Classifier           Output
─────────────────────           ─────────────           ──────
  SNR (dB)       ──────►                        ──────► BPSK   (1 bit/sym)
  BER            ──────►   Random Forest /       ──────► QPSK   (2 bits/sym)
                            SVM / MLP / GBM      ──────► 16-QAM (4 bits/sym)
                                                 ──────► 64-QAM (6 bits/sym)
```

The system learns the **decision boundaries** between modulation schemes from realistic channel data generated using standard BER–SNR formulas (Q-function, erfc), with Rayleigh fading noise applied for realism.

---

## ✨ Features

- 📊 **5 000-sample synthetic dataset** with realistic BER–SNR curves per modulation
- 🔧 **Feature engineering** — `log₁₀(BER)`, `SNR²`, `SNR × |log(BER)|`, and more
- 🤖 **4 ML classifiers** compared: Random Forest, Gradient Boosting, SVM (RBF), MLP
- 🔍 **GridSearchCV hyperparameter tuning** with 5-fold cross-validation
- 📡 **Real-time prediction function** returning modulation + confidence score
- 🌊 **500-step channel simulation** with time-varying fading
- 📈 **Rich visualisations** — confusion matrices, decision boundaries, BER curves, spectral efficiency plots
- 💾 **Model export** via `joblib` with metadata JSON

---

## 📐 Modulation Decision Logic

| SNR Range | Modulation | Bits / Symbol | Channel Condition |
|:---------:|:----------:|:-------------:|:-----------------:|
| < 5 dB    | **BPSK**   | 1             | Very noisy        |
| 5 – 10 dB | **QPSK**   | 2             | Moderate noise    |
| 10 – 20 dB| **16-QAM** | 4             | Good              |
| > 20 dB   | **64-QAM** | 6             | Excellent         |

> The ML model learns these boundaries from data rather than hard-coded thresholds, making it robust to non-ideal channel conditions.

---

## 🗂 Project Structure

```
ml-adaptive-modulation/
│
├── ML_Adaptive_Modulation_Francesco.ipynb   ← Main Colab notebook
│
├── adaptive_mod_model/                      ← Saved model artefacts
│   ├── model.pkl                            ← Trained sklearn pipeline
│   ├── label_encoder.pkl                    ← LabelEncoder for classes
│   └── metadata.json                        ← Model info & thresholds
│
├── README.md                                ← This file
└── LICENSE
```

---

## 🚀 Getting Started

### Option 1 — Google Colab (recommended)

1. Click the **Open in Colab** badge at the top of this page
2. Go to **Runtime → Run all**
3. All dependencies are installed automatically in the first cell

### Option 2 — Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/ml-adaptive-modulation.git
cd ml-adaptive-modulation

# Install dependencies
pip install numpy pandas matplotlib scikit-learn seaborn scipy joblib

# Launch Jupyter
jupyter notebook ML_Adaptive_Modulation_Francesco.ipynb
```

---

## 📓 Notebook Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Imports & Setup** | Install packages, set seeds |
| 2 | **BER–SNR Functions** | Q-function, erfc-based BER for all schemes |
| 3 | **Dataset Generation** | 5 000 synthetic samples with Rayleigh noise |
| 4 | **EDA** | Distributions, feature space, spectral efficiency |
| 5 | **Feature Engineering** | 5 engineered features + train/val/test split |
| 6 | **Model Training** | RF, GB, SVM, MLP — comparison table |
| 7 | **Evaluation** | Confusion matrix, F1 scores, accuracy chart |
| 8 | **Feature Importance** | Random Forest feature rankings |
| 9 | **Hyperparameter Tuning** | GridSearchCV with 5-fold CV |
| 10 | **Prediction Function** | `predict_modulation(snr_db, ber)` |
| 11 | **Channel Simulation** | 500-step fading channel with live switching |
| 12 | **Cross-Validation** | Violin plots, final summary table |
| 13 | **Model Export** | `joblib` save + reload verification |

---

## 📊 Results

### Accuracy Comparison

| Model | Val Accuracy | Test Accuracy | Train Time |
|-------|:------------:|:-------------:|:----------:|
| Random Forest     | ~0.98 | ~0.98 | Fast   |
| Gradient Boosting | ~0.97 | ~0.97 | Medium |
| SVM (RBF)         | ~0.97 | ~0.97 | Medium |
| MLP Neural Net    | ~0.96 | ~0.96 | Medium |

### Spectral Efficiency Gain

```
Fixed BPSK      →  1.00 bits/symbol  (baseline)
Adaptive (ML)   →  ~3.8 bits/symbol  (avg over simulated channel)
                   ─────────────────────────────
                   ~3.8× average gain | up to 6× at high SNR
```

---

## 🤖 Models Used

### Random Forest
An ensemble of decision trees that votes on the modulation class. Naturally handles the non-linear SNR/BER decision boundaries and provides feature importance rankings.

### Gradient Boosting
Sequential boosting of weak learners. Achieves strong accuracy with good generalisation through shrinkage and tree depth control.

### SVM with RBF Kernel
Maps features into a high-dimensional space to find the optimal separating hyperplane between modulation classes.

### MLP Neural Network
A 3-layer fully-connected network (`128 → 64 → 32`) with ReLU activations, trained via Adam with early stopping.

All models are wrapped in a **`sklearn.Pipeline`** with `StandardScaler` to ensure consistent preprocessing at inference time.

---

## 📦 Dependencies

```python
numpy >= 1.23
pandas >= 1.5
matplotlib >= 3.6
scikit-learn >= 1.1
seaborn >= 0.12
scipy >= 1.9
joblib >= 1.2
```

Install all at once:

```bash
pip install numpy pandas matplotlib scikit-learn seaborn scipy joblib
```

---

## 👤 Author

**Francesco De Florence**

> *Adaptive Modulation Using Machine Learning*  
> *Wireless Communications & Signal Processing*

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

*Built with Python · Scikit-Learn · Google Colab*

⭐ Star this repository if you found it useful!

</div>
