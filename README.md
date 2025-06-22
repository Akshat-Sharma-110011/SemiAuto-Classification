# SemiAuto-Classification

**SemiAuto-Classification** is a Python microservice for automating the machine learning lifecycle of classification tasks. It handles everything from data ingestion and preprocessing to model training, evaluation, and report generation. Designed for **ML engineers**, **MLOps practitioners**, and **software developers**, it exposes a REST API and a web-based dashboard for user-driven yet automated ML.

---

## 🚀 Features

- 🔍 **End-to-End ML Pipeline** — Covers data cleaning, preprocessing, training, evaluation, and reporting.
- ⚙️ **Semi-Automated Design** — User input + automated model selection and hyperparameter tuning.
- 🌐 **REST API + UI** — Trigger and monitor jobs via FastAPI (or Flask) endpoints and Jinja2 dashboard.
- 🧱 **Modular Architecture** — Swap out models, add transformations, or change training logic easily.
- 🐳 **Dockerized** — Fast setup and reproducible deployments via containerized builds.

---

## 🧱 Architecture Overview

```
SemiAuto-Classification/
├── app.py                      # Entry point for API / CLI
├── semiauto_classification/    # Core modules (training, preprocessing, reporting)
│   ├── data_preprocessing.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── ...
├── config/                     # Config files (intel.yaml, etc.)
├── data/                       # Raw / processed data
├── reports/                    # Output: metrics, plots, trained model
├── templates/                  # Jinja2 HTML templates for web UI
├── static/                     # CSS / JS for UI
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container setup
└── test_environment.py         # Dependency validation script
```

---

## ⚙️ Installation

### 🔧 Prerequisites
- Python 3.8+
- `pip`, `venv`, or Conda
- Docker (optional)

### 📦 Install via pip

```bash
pip install semiauto-classification
```

### 📥 Install from source

```bash
git clone https://github.com/Akshat-Sharma-110011/SemiAuto-Classification.git
cd SemiAuto-Classification
pip install -r requirements.txt
pip install -e .
```

### 🐳 Run via Docker

```bash
docker build -t semiauto-classification .
docker run -p 8000:8000 semiauto-classification
```

### ✅ Validate Installation

```bash
python check_environment.py
python test_environment.py
```

---

## 🔧 Usage

### 🖥️ Launch the Server

```bash
python app.py
```

### 🌐 Web Dashboard

Access at:
```
http://localhost:8000/
```

### 📡 API Endpoints

- `POST /train` — Trigger training  
- `POST /predict` — Make predictions  
- `GET /report` — Download evaluation report  

#### Example: Train Model via API

```bash
curl -X POST http://localhost:8000/train \
     -F file=@data.csv \
     -F target=LabelColumn
```

#### Example: Predict New Samples

```bash
curl -X POST http://localhost:8000/predict \
     -F file=@new_data.csv
```

### 📁 Outputs

- Trained models, confusion matrix, accuracy, precision/recall
- Saved under `reports/` directory

---

## 🛠️ Customization

- 🧩 **Plug in Models:** Extend `model_building.py` with new classifiers.
- 🧹 **Modify Preprocessing:** Edit `data_preprocessing.py` for custom transforms.
- ⚙️ **Configurable Inputs:** Adjust parameters in `intel.yaml` or CLI arguments.
- 🎨 **Update UI:** Customize Jinja2 templates in `templates/`.

---

## ⚠️ Limitations & Roadmap

- Currently supports **classification only**
- Basic AutoML (no feature engineering automation or advanced ensembling yet)
- No integrated monitoring or cloud support
- Plans include:
  - Multi-label support
  - Improved report UI
  - MLflow integration
  - Cloud-native & distributed extensions

---

## 📚 Credits & References

Inspired by the principles of AutoML frameworks such as:

- **[H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)** – For model leaderboard-based selection and automation
- **[AutoML: A Survey of the State-of-the-Art](https://arxiv.org/abs/1904.12054)** – Hutter et al., 2019

> “Automated Machine Learning (AutoML) aims to automate the end-to-end process of applying machine learning to real-world problems.” – Hutter et al.

---

## 🪪 License

Licensed under the [MIT License](LICENSE).
