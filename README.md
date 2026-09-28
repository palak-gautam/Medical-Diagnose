# Medical Diagnose

### Multimodal Machine Learning System for Preliminary Disease Prediction

> **Medical Diagnose** is a multimodal machine-learning research project that combines **symptom information and medical images** for preliminary disease prediction. It evaluates different modelling and fusion configurations through machine learning, deep learning, and controlled experimentation.

---

## Research Snapshot

|                             |                               |
| --------------------------- | ----------------------------- |
| **Project Type**            | Research / Machine Learning   |
| **Research Area**           | Multimodal disease prediction |
| **Input Modalities**        | Symptom text + medical images |
| **Text Representation**     | TF-IDF                        |
| **Image Model**             | EfficientNetB0                |
| **Deep Learning Framework** | PyTorch                       |
| **ML Baseline**             | Random Forest                 |
| **Web Application**         | Flask                         |
| **Best Evaluated Variant**  | A7                            |
| **Top-1 Accuracy**          | **87.00%**                    |
| **Top-3 Accuracy**          | **97.14%**                    |
| **F1 Score**                | **0.8707**                    |

---

# 1. Overview

Medical Diagnose combines **textual symptom information and medical images** within a multimodal prediction pipeline.

Each modality is processed separately before being incorporated into the modelling workflow. Multiple configurations are evaluated through a controlled **ablation study** to understand how individual components and fusion strategies affect performance.

---

# 2. System Architecture

The system accepts symptom information and an optional medical image:

```text
                    Patient Input
                    /           \
                   /             \
          Symptom Information    Medical Image
                  ↓                    ↓
             NLP Pipeline       CNN Pipeline
                  ↓                    ↓
             Text Features       Image Features
                   \                  /
                    \                /
                     ↓              ↓
                    Multimodal
                    Processing
                         ↓
                   Classification
                         ↓
                 Disease Prediction
```

### Multimodal Processing

```text
Symptom Text
     ↓
Text Preprocessing
     ↓
TF-IDF Representation
     ↓
Text Features
                    \
                     → Multimodal Model → Classifier → Prediction
                    /
Medical Image
     ↓
Image Preprocessing
     ↓
EfficientNetB0
     ↓
Image Features
```

---

# 3. Methodology

## 3.1 Text Processing

Symptom information is converted into numerical features using **TF-IDF (Term Frequency–Inverse Document Frequency)**.

```text
Symptom Description
        ↓
Text Preprocessing
        ↓
TF-IDF Vectorization
        ↓
Numerical Feature Representation
        ↓
NLP Component
```

## 3.2 Image Processing

Medical images are processed using **EfficientNetB0**, a CNN architecture with ImageNet-pretrained weights.

```text
Medical Image
      ↓
Image Preprocessing
      ↓
EfficientNetB0
      ↓
Visual Feature Extraction
      ↓
Image Representation
```

The resulting text and image representations are used within the multimodal modelling pipeline.

---

# 4. PEPA — Performance Evaluation Process Algebra

**PEPA stands for Performance Evaluation Process Algebra.**

PEPA is a formal language and mathematical framework for modelling and analysing systems composed of multiple interacting components. It was developed for performance analysis and has also been adapted to biological systems, where interacting biochemical processes can be represented and studied over time.

### Why PEPA Is Relevant

Complex biological systems involve multiple interacting components whose behaviour can evolve over time. Process algebra provides a formal way to represent:

* interacting components
* system states
* state transitions
* component cooperation
* system-level behaviour
* temporal evolution

```text
                    Complex System
                         │
              ┌──────────┴──────────┐
              ↓                     ↓
         Component A           Component B
              │                     │
              └────── Interaction ──┘
                         ↓
                 System Behaviour
                         ↓
                  Model / Analysis
```

Within Medical Diagnose, PEPA is considered as part of the project's broader approach to modelling complex interacting systems.

---

# 5. Experimental Design

The project evaluates multiple modelling configurations through an **ablation study** to measure the contribution of different components.

| Variant | Configuration                    |
| ------- | -------------------------------- |
| **A1**  | Random Forest baseline           |
| **A2**  | CNN encoder only                 |
| **A3**  | NLP encoder only                 |
| **A4**  | CNN + NLP concatenation          |
| **A5**  | CNN + NLP mean fusion            |
| **A6**  | CNN + NLP + PEPA without dropout |
| **A7**  | CNN + NLP + PEPA with dropout    |

This comparison covers individual modalities, conventional multimodal approaches, and the proposed configuration.

---

# 6. Ablation Study

| Variant | Configuration                | Top-1 Accuracy | Top-3 Accuracy |   F1 Score |
| ------- | ---------------------------- | -------------: | -------------: | ---------: |
| A1      | Random Forest baseline       |         80.31% |         92.98% |     0.8116 |
| A2      | CNN encoder only             |         67.80% |         86.30% |     0.6544 |
| A3      | NLP encoder only             |         85.27% |         96.50% |     0.8546 |
| A4      | CNN + NLP concatenation      |         86.70% |         96.81% |     0.8677 |
| A5      | CNN + NLP mean fusion        |         86.75% |         96.98% |     0.8691 |
| A6      | CNN + NLP + PEPA, no dropout |         86.95% |         97.14% |     0.8696 |
| **A7**  | **CNN + NLP + PEPA**         |     **87.00%** |     **97.14%** | **0.8707** |

### Experimental Observation

The reported experiments show that:

* NLP-only modelling outperformed CNN-only modelling in this evaluation.
* Multimodal configurations improved over the individual CNN configuration.
* PEPA-based configurations produced the highest reported F1 scores.
* A7 achieved **87.00% Top-1 accuracy, 97.14% Top-3 accuracy, and 0.8707 F1**.

These results reflect the project's experimental setup and do not represent clinical validation.

---

# 7. Results

## Proposed Configuration — A7

| Metric             |     Result |
| ------------------ | ---------: |
| **Top-1 Accuracy** | **87.00%** |
| **Top-3 Accuracy** | **97.14%** |
| **F1 Score**       | **0.8707** |

### Top-1 Accuracy Comparison

```text
A1  Random Forest             80.31%
A2  CNN only                  67.80%
A3  NLP only                  85.27%
A4  CNN + NLP                 86.70%
A5  Mean Fusion               86.75%
A6  PEPA, no dropout          86.95%
A7  PEPA                      87.00%
```

---

# 8. Dataset Composition

The project uses publicly available datasets containing symptom information and medical images.

## Text Datasets

| Dataset                  | Source             |      Rows | Diseases |
| ------------------------ | ------------------ | --------: | -------: |
| DDXPlus                  | Figshare / Mila AI | 1,307,508 |       49 |
| Disease-Symptom Severity | Kaggle             |     4,920 |       41 |
| Patient Symptom Profile  | Kaggle             |   250,000 |       14 |
| Symptoms to Diseases     | Kaggle             |  Variable |      713 |

## Image Datasets

| Category            | Dataset Group                                      | Images / Volumes |
| ------------------- | -------------------------------------------------- | ---------------: |
| Chest & Respiratory | COVID-19 Radiography, NIH Chest X-ray 14, Chest CT |          134,285 |
| Neurological        | Brain Tumor MRI, Alzheimer MRI, Brain Stroke CT    |           15,923 |
| Skin                | ISIC 2019, HAM10000, DermNet, Skin Disease         |           67,846 |
| Eye                 | Eye Disease Classification, Retinal OCT            |           88,712 |
| Heart               | ECG Heartbeat                                      |          109,446 |
| Throat              | Thyroid Ultrasound                                 |              637 |
| Bone & Joint        | Knee Osteoarthritis, MURA Bone X-ray               |           50,681 |
| Abdominal           | Liver CT Scan                                      |      131 volumes |
| Kidney              | Kidney CT Scan                                     |           12,446 |
| Infection           | Malaria Cell Images                                |           27,558 |

---

# 9. Project Structure

```text
Medical-Diagnose/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── dataset/
│   ├── raw/
│   │   ├── text/
│   │   └── images/
│   └── cleaned/
│       ├── text/
│       └── images/
│
├── notebooks/
│   ├── clean_text_datasets.ipynb
│   ├── clean_image_datasets.ipynb
│   ├── nlp_encoder.ipynb
│   ├── cnn_encoder.ipynb
│   ├── pepa_module.ipynb
│   ├── pipeline.ipynb
│   ├── train.ipynb
│   └── evaluate.ipynb
│
├── artifacts/
│   ├── model_best.pth
│   ├── model_config.pkl
│   ├── encoder.pkl
│   ├── vectorizer.pkl
│   └── symptom_cols.pkl
│
├── results/
│   ├── ablation_results.csv
│   └── figures/
│
└── webapp/
    ├── app.py
    ├── templates/
    └── static/
```

---

# 10. Web Application

The trained model is integrated into a **Flask web application**, connecting the research pipeline with a user-facing inference workflow.

```text
User Input
    ↓
Flask Application
    ↓
Symptom + Image Processing
    ↓
Feature Extraction
    ↓
Trained Model
    ↓
Disease Prediction
    ↓
Result Interface
```

This provides an end-to-end demonstration from patient input to model inference.

---

# 11. Reproducibility

Install the project dependencies:

```bash
pip install -r requirements.txt
```

For the CUDA-enabled PyTorch environment:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU availability:

```python
import torch

print(torch.cuda.is_available())

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Run the preprocessing, training, and evaluation notebooks according to the repository workflow.

Launch the Flask application:

```bash
cd webapp
python app.py
```

---

# 12. Limitations

Medical Diagnose is a **research and educational machine-learning prototype**, not a clinically validated diagnostic system.

* Public datasets may not represent real-world clinical populations.
* Dataset quality and label consistency can affect performance.
* Results depend on the experimental dataset and evaluation setup.
* Independent external validation would be required for clinical interpretation.
* The system should not be used for real-world medical decision-making.

---

# 13. Future Research

* Explainable AI for model predictions
* More advanced NLP representations
* Multilingual symptom understanding
* Additional and more diverse datasets
* Improved multimodal modelling
* Process-algebraic modelling of complex biological interactions
* Mobile deployment
* Independent external evaluation

---

# 14. Technology Stack

| Area                 | Technologies                                  |
| -------------------- | --------------------------------------------- |
| **Deep Learning**    | PyTorch                                       |
| **Computer Vision**  | EfficientNetB0 · torchvision · Pillow         |
| **NLP**              | TF-IDF                                        |
| **Machine Learning** | scikit-learn · Random Forest                  |
| **Formal Modelling** | PEPA — Performance Evaluation Process Algebra |
| **Web**              | Flask                                         |
| **Data Processing**  | Pandas · NumPy                                |
| **Experimentation**  | Jupyter Notebook                              |
| **Acceleration**     | CUDA                                          |

---

# 15. Medical Disclaimer

> **Medical Diagnose is a research and educational prototype.**
>
> It is not a medical device and should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.
>
> Predictions generated by the system should not be interpreted as medical advice. Real-world medical decisions should be made by qualified healthcare professionals.

---

## Research Perspective

Medical Diagnose combines **machine learning, deep learning, multimodal data, and formal system modelling** within one research project.

The project evaluates how heterogeneous medical information can be incorporated into disease prediction while using **Performance Evaluation Process Algebra (PEPA)** as a formal modelling perspective for complex interacting systems.

A controlled **ablation study** provides experimental evidence for comparing the evaluated modelling configurations.

The project therefore combines **implementation, experimentation, and formal modelling** rather than presenting only a conventional ML application.
