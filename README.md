# 🌊 Urban Flood Early Warning System (UFEWS)
### Enterprise-Grade MLOps Platform for Disaster Risk Prediction

[![CI Pipeline](https://img.shields.io/badge/CI-GitHub%20Actions-blue?style=for-the-badge&logo=githubactions)](https://github.com/VIVEK-MARRI/urban_flood_early_warning/actions)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker)](https://www.docker.com/)
[![Code Style](https://img.shields.io/badge/Code%20Style-Black-000000?style=for-the-badge)](https://github.com/psf/black)

---

## 📖 Project Overview
The **Urban Flood Early Warning System** is a production-ready MLOps solution designed to predict flood risks using real-time weather data. It demonstrates a complete end-to-end machine learning lifecycle, from data ingestion and versioning to model deployment and monitoring.

This project is built to simulate a real-world enterprise environment, ensuring **reproducibility**, **scalability**, and **observability**.

---

## 🏗 System Architecture

The following diagram illustrates the high-level architecture and data flow of the system.

```mermaid
graph LR
    subgraph Data_Layer ["Data Ops"]
        direction TB
        Raw["Raw Weather Data"] -->|DVC Tracking| DVC_Store[("DVC Local Storage")]
        DVC_Store -->|dvc pull| Pipeline
    end

    subgraph Pipeline ["Orchestration (Airflow)"]
        direction TB
        Ingest["Data Ingestion"] --> Preprocess[Preprocessing]
        Preprocess --> Train["XGBoost Training"]
        Train --> Evaluate["Model Evaluation"]
        Evaluate -->|Threshold Pass| Register["Register Model"]
    end

    subgraph Experimentation ["MLflow Platform"]
        Train -.->|Log Params & Metrics| Tracking["Tracking Server"]
        Register -.->|Version Control| Registry["Model Registry"]
    end

    subgraph Serving ["Production Serving"]
        Registry -->|Load Version| API["FastAPI Service"]
        API <-->|REST| Dashboard["Streamlit UI"]
    end

    subgraph Observability ["Monitoring Stack"]
        API -.->|Expose Metrics| Prometheus
        Prometheus -->|Visualize| Grafana["Grafana Dashboards"]
    end

    Data_Layer --> Pipeline
```

---

## 🛠 Technology Stack

| Domain | Technology | Purpose |
| :--- | :--- | :--- |
| **Orchestration** | ![Airflow](https://img.shields.io/badge/Apache%20Airflow-017EEO?style=flat-square&logo=Apache%20Airflow&logoColor=white) | Automates data pipelines and retraining schedules. |
| **Tracking** | ![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat-square&logo=MLflow&logoColor=white) | Manages experiment tracking and model registry. |
| **Version Control** | ![DVC](https://img.shields.io/badge/DVC-945DD6?style=flat-square&logo=data-version-control&logoColor=white) | Versions large datasets alongside code. |
| **Serving** | ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | High-performance API for real-time inference. |
| **Frontend** | ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white) | Interactive dashboard for end-users. |
| **Monitoring** | ![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=flat-square&logo=Prometheus&logoColor=white) | Collects system and model performance metrics. |
| **Visualization** | ![Grafana](https://img.shields.io/badge/Grafana-F46800?style=flat-square&logo=Grafana&logoColor=white) | Visualizes metrics and sets up alerts. |
| **Containerization** | ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=Docker&logoColor=white) | Ensures consistent environments across dev and prod. |

---

## 🚀 Getting Started

### Prerequisites
*   **Docker Desktop** (Engine 20.10+)
*   **Python 3.9+** (For local development)
*   **Git**

### ⚡ Quick Start (Docker Composition)
Deploy the entire MLOps stack with a single command.

```bash
# 1. Clone the repository
git clone https://github.com/VIVEK-MARRI/urban_flood_early_warning.git
cd urban_flood_early_warning

# 2. Start the application (Background mode)
make up
```

Access the services:
*   🌊 **Streamlit Dashboard**: [http://localhost:8501](http://localhost:8501)
*   🌬 **Airflow**: [http://localhost:8081](http://localhost:8081) (`admin` / `admin`)
*   🧪 **MLflow**: [http://localhost:5000](http://localhost:5000)
*   ⚡ **FastAPI**: [http://localhost:8000/docs](http://localhost:8000/docs)
*   📊 **Grafana**: [http://localhost:3000](http://localhost:3000) (`admin` / `admin`)

---

## 👩‍💻 Local Development

For developers contributing to the codebase:

1.  **Environment Setup**:
    ```bash
    make setup
    # Installs dev dependencies (pytest, black, dvc, etc.)
    ```

2.  **Data Management (DVC)**:
    ```bash
    dvc pull
    # Downloads the latest dataset from local/remote storage
    ```

3.  **Running Tests**:
    ```bash
    make test
    # Runs unit tests with mocked MLflow connections
    ```

4.  **Code Quality**:
    ```bash
    make lint
    # Enforces Black formatting
    ```

---

## 🤖 CI/CD Pipeline

We use **GitHub Actions** to enforce quality standards.
*   **Trigger**: Pushes to `main`.
*   **Job**: `build-and-test`.
*   **Steps**:
    1.  Checkout Code.
    2.  Install Dependencies.
    3.  **Linting**: Block merge if code style violations are found.
    4.  **Testing**: Run `pytest` suite.

---

## 📂 Project Structure

```bash
urban-flood-system/
├── airflow/            # DAGs and pipeline logic
├── api/                # REST API implementation
├── data/               # DVC-tracked datasets
├── models/             # Model artifacts
├── monitoring/         # Observability configurations
├── notebooks/          # EDA and research
├── src/                # Core utility packages
├── streamlit_app/      # Frontend application
└── tests/              # Automated test suite
```

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details..