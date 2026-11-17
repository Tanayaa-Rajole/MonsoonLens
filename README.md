# 🌧️ MonsoonLens
### An Explainable AI-Powered Rainfall Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange.svg)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

**MonsoonLens** is a machine learning-based rainfall intelligence platform designed to analyse historical rainfall patterns, estimate next-day rainfall and detect unusual precipitation behaviour across Indian districts.

The project combines data science, predictive modelling, statistical anomaly detection and interactive visualisation to make historical rainfall data more accessible and interpretable.

---

## 📌 Project Overview

Rainfall patterns play a significant role in agriculture, water resource management, infrastructure planning and environmental monitoring.

However, analysing historical rainfall records and identifying unusual precipitation patterns requires processing large datasets and interpreting complex trends.

MonsoonLens aims to simplify this process through an interactive platform that transforms historical rainfall observations into meaningful analytical insights.

### 🎯 Objectives

- Analyse historical rainfall patterns across Indian districts.
- Develop machine learning models for next-day rainfall estimation.
- Identify unusual rainfall patterns using statistical anomaly detection.
- Compare the performance of different predictive models.
- Visualise rainfall trends through interactive dashboards.
- Improve model interpretability through feature importance analysis.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| Rainfall Explorer | Explore historical rainfall trends and seasonal patterns. |
| Rainfall Prediction | Estimate next-day rainfall using machine learning. |
| Anomaly Detection | Identify unusually high or low rainfall observations. |
| Regional Comparison | Compare rainfall distributions across districts. |
| Model Evaluation | Evaluate predictive performance using regression metrics. |
| Explainable ML | Understand the contribution of input features to predictions. |

---

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Processing | Pandas, NumPy |
| Data Visualisation | Matplotlib, Seaborn, Plotly |
| Machine Learning | Scikit-learn |
| Advanced Modelling | XGBoost (planned) |
| Anomaly Detection | Statistical methods |
| Dashboard | Streamlit |
| Model Persistence | Joblib |
| Testing | Pytest |
| Version Control | Git, GitHub |

---

## 🧠 Machine Learning Approach

MonsoonLens follows a structured machine learning pipeline.

**1. Data Collection**

Acquire historical rainfall observations from publicly available meteorological datasets.

**2. Data Preprocessing**

- Handle missing observations.
- Convert and validate date fields.
- Remove invalid records.
- Prepare district-level datasets.

**3. Feature Engineering**

Generate meaningful features, including:

- Previous-day rainfall.
- Historical rainfall lag values.
- Rolling rainfall averages.
- Monthly and seasonal indicators.
- Other available historical weather features.

**4. Model Development**

The initial implementation will compare:

- Mean baseline.
- Linear Regression.
- Random Forest Regressor.
- XGBoost Regressor (optional).

**5. Model Evaluation**

Models will be evaluated using:

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- Coefficient of Determination (R²).

Chronological train-test splitting will be used to reduce temporal data leakage.

**6. Anomaly Detection**

Historical rainfall distributions will be analysed to identify observations that deviate significantly from seasonal and regional patterns.

**7. Explainability**

Feature importance and prediction analysis will help communicate how the model produces its estimates.

---

## 🏗️ Project Architecture

```text
                  Historical Rainfall Data
                            |
                            v
                    Data Collection
                            |
                            v
                    Data Preprocessing
                            |
                            v
                    Feature Engineering
                            |
               +------------+------------+
               |                         |
               v                         v
        Prediction Models         Anomaly Detection
               |                         |
               v                         v
         Model Evaluation          Statistical Analysis
               |                         |
               +------------+------------+
                            |
                            v
                    Explainability
                            |
                            v
                   Streamlit Dashboard
                            |
                            v
                 Interactive Insights
```

---

## 📂 Repository Structure

```text
MonsoonLens/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   └── 03_model_experiments.ipynb
│
├── models/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   └── anomaly_detection.py
│
├── app/
│   ├── Home.py
│   └── pages/
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_features.py
│
├── reports/
│   └── model_results.md
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📊 Dataset Sources

The project intends to use publicly available rainfall datasets.

| Dataset | Purpose | Source |
|---|---|---|
| IMD Rainfall Data | Primary historical rainfall observations | [India Meteorological Department](https://www.imdpune.gov.in/) |
| National Water Data Portal | District-wise rainfall data discovery | [India-WRIS](https://indiawris.gov.in/) |
| CHIRPS | Additional rainfall data for comparative analysis | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) |

Dataset selection, temporal coverage, spatial resolution and licensing will be documented during implementation.

**Note:** Dataset sources are listed as intended sources. Actual datasets and their coverage will be documented after acquisition and validation.

---

## 🚀 Installation and Setup

### Prerequisites

- Python 3.10 or above.
- Git.
- VS Code or another Python-compatible editor.

### 1. Clone the repository

```bash
git clone https://github.com/Tanayaa-Rajole/MonsoonLens.git
```

### 2. Navigate to the project directory

```bash
cd MonsoonLens
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
streamlit run app/Home.py
```

---

## 📈 Evaluation Metrics

The predictive models will be evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| MAE | Measures average absolute prediction error. |
| RMSE | Penalises larger prediction errors. |
| R² Score | Measures the proportion of variance explained by the model. |

Actual model performance results will be added after training and evaluation.

---

## 🗺️ Development Roadmap

- [ ] Acquire and validate historical rainfall datasets.
- [ ] Implement data preprocessing pipeline.
- [ ] Perform exploratory data analysis.
- [ ] Develop rainfall-related features.
- [ ] Implement baseline regression models.
- [ ] Train and evaluate Random Forest.
- [ ] Implement rainfall anomaly detection.
- [ ] Add feature importance analysis.
- [ ] Develop interactive Streamlit dashboard.
- [ ] Integrate all application modules.
- [ ] Implement automated tests.
- [ ] Deploy the application.
- [ ] Add final performance results and documentation.

---

## 🔮 Future Enhancements

Potential future improvements include:

- Geospatial rainfall visualisation.
- Satellite-based rainfall data integration.
- LSTM-based rainfall modelling.
- SHAP-based model explanations.
- Comparative analysis of multiple rainfall datasets.
- Expanded district-level historical trend analysis.

---

## ⚠️ Limitations

- Predictions depend on the quality and availability of historical observations.
- Historical rainfall patterns do not guarantee future weather conditions.
- Model performance may vary across districts and seasons.
- Anomaly detection identifies statistical deviations, not necessarily meteorological emergencies.
- The platform is intended for educational and analytical purposes, not operational weather forecasting or disaster-warning decisions.

---

## 👩‍💻 Author

**Tanayaa**

Engineering Student | Electronics and Telecommunication Engineering

Interested in Machine Learning, Artificial Intelligence, Data Science and intelligent software systems.

---

## 📄 License

This project is intended to be released under the MIT License. Dataset terms and attribution requirements remain applicable independently of the project license.

---

⭐ If you find this project interesting, consider starring the repository!

**MonsoonLens: Turning rainfall data into meaningful intelligence.**
