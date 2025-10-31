# 🌊 Urban Flood Early Warning System — End-to-End AI & MLOps Project

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%7C%20Streamlit-green)
![MLOps](https://img.shields.io/badge/MLOps-AWS%20%7C%20Docker%20%7C%20GitHub%20Actions-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)

---

## 🧭 Overview

The **Urban Flood Early Warning System** is an end-to-end **AI + MLOps project** that predicts flood risk in urban areas using **real-time sensor data** and **machine learning pipelines**.  
It demonstrates how **data science, ML, cloud infrastructure, and CI/CD automation** work together to build a **production-ready early warning platform** for urban disaster management.

---

## 🎯 Objectives

- Predict **urban flood occurrences** using machine learning.
- Process **real-time IoT sensor data** (rainfall, water level, humidity, etc.).
- Automate **data pipelines**, **model retraining**, and **deployment** using MLOps.
- Provide a **Streamlit dashboard** for real-time flood monitoring.
- Deploy **FastAPI backend** on AWS with **CI/CD automation**.

---

## 🧱 System Architecture

```
      ┌───────────────────────────┐
      │   Continuous Data Source  │
      │ (Simulated Live Weather)  │
      └──────────────┬────────────┘
                     │
                     ▼
      ┌──────────────────────────────────┐
      │ Apache Airflow (Orchestration)   │
      │ 1. Data Ingestion & Feature Eng. │
      │ 2. Prediction Run (Every 5 min)  │
      └────────────┬─────────────────────┘
                   │
    (Prediction Data)
                   ▼
      ┌───────────────────────────────────┐
      │ PostgreSQL (Transactional Audit)  │
      │  (Logging Prediction & Telemetry) │
      └────────────┬──────────────┬───────┘
                   │              │
                   ▼              ▼
  (For Monitoring) (For Retraining, Triggered by Schedule)
      ┌─────────────────┐ ┌──────────────────────────┐
      │ Streamlit Monitor & Alerts │ │   ML Model Training (XGBoost)  │
      └─────────────────┘ └────────────┬─────────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ MLflow Governance & Deployment    │
                     │ (Promote to 'Production' Registry)│
                     └────────────┬──────────────────────┘
                                  │
                                  ▼
      ┌──────────────────────────────────┐
      │ FastAPI Inference Service        │
      │ (Loads Production Model & Serves)│
      └──────────────────────────────────┘
```
```

---

## 📁 Project Structure

```text
urban_flood_early_warning/
│
├── data/                        # Raw, processed, and prediction logs
│   ├── raw/
│   ├── processed/
│   └── prediction_log.csv
│
├── src/                         # Core ML pipeline modules
│   ├── components/              # Ingestion, Validation, Transformation
│   ├── pipeline/                # Training & Prediction pipelines
│   ├── utils/                   # Helper functions & utilities
│   └── entity/                  # Configurations & artifacts
│
├── app.py                       # FastAPI app for model inference
├── dashboard.py                 # Streamlit dashboard for monitoring
├── Dockerfile                   # Docker container setup
├── requirements.txt             # Dependencies list
├── .github/workflows/ci.yml     # GitHub Actions CI/CD pipeline
└── README.md                    # Documentation (you are here)
```

---

## 🧠 Tech Stack

| **Category** | **Tools & Technologies** | **Reason/Proof in Project** |
| :--- | :--- | :--- |
| **Language** | **Python 3.10** | Standard for data processing and serving layers. |
| **Orchestration** | **Apache Airflow** | Controls CI/CD (retraining) and continuous prediction loops. |
| **Serving Layer** | **FastAPI** | High-performance, low-latency asynchronous API serving. |
| **Front-End/Monitoring** | **Streamlit, Plotly** | Dashboard and interactive geospatial map visualization. |
| **Database/Audit** | **PostgreSQL** | **Transactional logging (ACID)** for all prediction data. |
| **Modeling** | **XGBoost, Scikit-learn** | XGBoost (promoted algorithm) and Scikit-learn for calibration/metrics. |
| **MLOps Governance** | **MLflow (Tracking & Registry)** | Manages model versions and production deployment staging. |
| **Containerization** | **Docker, Docker Compose** | Guarantees reproducible environments across all services. |
| **Metrics & Observability** | **Prometheus** | Tracks real-time API latency and request/prediction distribution. |

---

## 🔁 ML Workflow

### 1️⃣ Data Ingestion  
- Fetches historical & real-time data from **MongoDB** and local sources.  
- Cleans and merges multiple data streams into one consistent schema.  

### 2️⃣ Data Validation  
- Validates schema using **schema.yaml** to ensure consistency.  
- Handles missing values, duplicates, and outliers.  

### 3️⃣ Data Transformation  
- Performs **feature scaling**, encoding, and feature engineering.  
- Creates **train-test splits** and stores transformed data artifacts.  

### 4️⃣ Model Training  
- Trains ensemble models (**RandomForest**, **XGBoost**) for flood risk prediction.  
- Tracks metrics with **MLflow** (Accuracy, Precision, Recall, F1-Score).  

### 5️⃣ Model Evaluation  
- Compares newly trained models against production models.  
- Automatically pushes the **best-performing model** to **AWS S3**.  

### 6️⃣ Model Deployment  
- Deploys using **FastAPI** for real-time inference.  
- Visualized via **Streamlit dashboard** for flood monitoring and analytics.  

### 7️⃣ MLOps Automation  
- **CI/CD pipeline** automates build → test → deploy using **GitHub Actions**.  
- **Docker** ensures consistent runtime environment across development and production.

---

## 🌦️ Streamlit Dashboard Features

- 📊 **Real-time flood risk visualization** — Displays live predictions and alerts from the deployed model.  
- 🌍 **Location-based flood probability mapping** — Interactive map showing regional flood probabilities.  
- 📈 **Model performance metrics dashboard** — Visualizes metrics like Accuracy, Precision, Recall, and F1-Score.  
- 🔁 **Auto-refresh prediction logs** — Automatically updates logs to reflect the latest predictions.  
- ⚙️ **Interactive flood scenario simulation** — Allows users to simulate various rainfall and terrain conditions to observe predicted outcomes.  

---

## ⚙️ FastAPI Endpoints

| **Endpoint**       | **Method** | **Description**                        |
|--------------------|------------|----------------------------------------|
| `/predict_flood`   | POST       | Predict flood risk from sensor data    |
| `/health`          | GET        | API health check                       |
| `/logs`            | GET        | Retrieve latest prediction logs        |

### 🧩 Example Request

```json
{
  "rainfall": 45.2,
  "humidity": 80.1,
  "temperature": 28.3,
  "water_level": 4.5,
  "soil_moisture": 70.0
}
```

### 🧾 Example Response

```json
{
  "flood_risk": "High",
  "probability": 0.87
}
```

---

## ⚙️ Setup & Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/VIVEK-MARRI/urban_flood_early_warning.git
cd urban_flood_early_warning
```

### 2️⃣ Create Environment

```bash
conda create -n floodenv python=3.10 -y
conda activate floodenv
pip install -r requirements.txt
```

### 3️⃣ Configure Database

Update your `.env` file:

```ini
POSTGRES_USER=postgres
POSTGRES_PASSWORD=yourpassword
POSTGRES_DB=flood_db
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

### 4️⃣ Launch Services (Docker Compose)

```bash
docker-compose up --build
```

This will start:

- Airflow Scheduler + Webserver
- PostgreSQL Database
- FastAPI Backend
- Streamlit Dashboard

---

## 🚀 Airflow Pipeline Overview

| **DAG Name**         | **Purpose**                    | **Schedule** |
|----------------------|--------------------------------|--------------|
| data_ingestion_dag   | Load & clean raw IoT data      | @hourly      |
| model_training_dag   | Train models & log metrics     | @daily       |
| model_retrain_dag    | Retrain when drift detected    | @weekly      |

All DAGs are managed in `airflow/dags/`.

---

## 🧮 Model Explainability & Monitoring

- **SHAP Values** → Explain which features influence flood risk.
- **MLflow** → Tracks model versions, hyperparameters, and metrics.
- **Data Drift Detection** → Automatically retrains when distribution changes.
- **PostgreSQL Logs** → Ensures transactional data consistency.

---

## 📊 Key Achievements

✅ Achieved >90% accuracy in flood risk prediction.  
✅ Built fully containerized Airflow + FastAPI + Streamlit ecosystem.  
✅ Automated ETL → Model → Deployment through Airflow.  
✅ Integrated real-time dashboards with clean UX.  
✅ Ensured data integrity & reproducibility with PostgreSQL and MLflow.  

---

## ☁️ Deployment Summary

| **Component**   | **Service**       |
|-----------------|-------------------|
| Backend API     | FastAPI           |
| Dashboard       | Streamlit         |
| Database        | PostgreSQL        |
| Orchestration   | Apache Airflow    |
| Containerization| Docker Compose    |
| Model Tracking  | MLflow            |

---

## 🔮 Future Enhancements

- 🌦️ Integrate satellite imagery (Sentinel) for spatial flood mapping.
- 📡 Add live rainfall API feeds for better real-time predictions.
- 🧠 Incorporate LSTM/GRU deep learning models for time-series forecasting.
- ☁️ Deploy on Kubernetes (EKS/GKE) for scaling.
- 📩 Add email/SMS alerts for high-risk notifications.

---

## 👨‍💻 Author

**Vivek Marri**  
💼 AI/ML & MLOps Engineer  
📧 Email: vivekyadavmarri.com  
🌐 [LinkedIn](https://www.linkedin.com/in/vivek-marri-49419a274/)  
🐙 [GitHub](https://github.com/VIVEK-MARRI)  

---

## 🧾 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🏁 Final Summary

The Urban Flood Early Warning System exemplifies a production-grade AI + MLOps ecosystem — integrating data pipelines, model automation, and deployment into one seamless workflow.  
It represents the bridge between AI research and operational reliability, showcasing real-world disaster management powered by MLOps excellence.
