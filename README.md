# Rakuten Product Classification - MLOps Pipeline

This project demonstrates a complete MLOps pipeline for the [Rakuten product classification challenge](https://challengedata.ens.fr/participants/challenges/35/), focusing on deployment, versioning, and operational aspects rather than model accuracy.

## Project Overview

- **Task**: Classify products into categories using text and image data
- **Focus**: MLOps infrastructure (MLflow, FastAPI, Docker, model versioning)
- **Models**: Classical ML pipeline with XGBoost, Random Forest, Logistic Regression, and SVM (with plans to integrate deep learning models later)

## Project Structure

```
rakuten_project/
├── config/                  # Configuration files
│   └── airflow.cfg         # Airflow configuration
├── scripts/                 # Setup and data loading scripts
│   ├── 1_download.sh       # Downloads Rakuten dataset
│   ├── 2_unzip_install.sh  # Extracts and installs
│   └── 3_service_load.sh   # Loads data into PostgreSQL + MinIO
├── src/                     # Source code
│   ├── load_minio.py       # MinIO object storage operations
│   ├── load_postgres.py    # PostgreSQL database operations
│   └── (ML pipeline components to be added)
├── .env                    # Environment variables
├── .gitignore
├── docker-compose.yml      # Docker services configuration
├── docker-compose.backup.yaml  # Backup Docker configuration
├── requirements.txt
├── servers.json           # Database server configuration
└── README.md              # This file
```

**Note:** Data is stored in Docker containers (PostgreSQL for metadata, MinIO for images), not in local folders.

## Setup Instructions

### 1. Clone Repository
```bash
git clone https://github.com/Pockyee/rakuten_project.git
cd rakuten_project
```

### 2. Setup Infrastructure, Environment
Follow the automated setup process:

1. **Make scripts executable:**
   ```bash
   chmod +x ./scripts/*
   ```

2. **Run setup scripts in order:**
   ```bash
   ./scripts/1_install_docker.sh      # Install Docker and unzip utility
   ./scripts/2_download.sh            # Download datasets and unzip images package
   ./scripts/3_run_docker.sh          # Start Docker containers
   ```
   *Some machines may not be able to start all Docker containers simultaneously due to performance limitations. Please rerun 3_run_docker.sh or directly run `docker compose up -d` in the project root directory.*

3. **Access services:**

- **Airflow** (Workflow GUI): [http://localhost:8080](http://localhost:8080)  
  - **Username:** `airflow`  
  - **Password:** `airflow`

- **pgAdmin** (database GUI): [http://localhost:8081](http://localhost:8081)  
  - **Email:** `rakuten@admin.com`  
  - **Password:** `rakutenadmin`

- **MinIO** (object storage GUI): [http://localhost:9001](http://localhost:9001)  
  - **Username:** `rakutenadmin`  
  - **Password:** `rakutenadmin`
   - *If accessing from another machine, replace localhost with your server's IP address*

**Note:** The scripts handle - system packages, Docker installation, and data loading.

### 3. Load data

Enter the Airflow UI and run the prepare_data DAG. After it completes successfully, the XCom under load_test_image should return the value 2762, and the XCom under load_train_image should return the value 16983. By default, 20% of the data is loaded into the database.

### 4. Run the Application
```bash
# Start the API
python -m uvicorn src.api.main:app --reload

# Run MLflow UI
mlflow ui
```

## MLOps Metrics

The following metrics will be tracked and displayed in a Grafana Dashboard (model performance is tracked in MLflow for information only):

**Data Quality Monitoring:**
- Outliers ratio
- Null values ratio  
- Basic statistics
- Unbalance ratio
- Suspicious correlations

**Data Drift Monitoring:**
- Jensen–Shannon Distance
- Population Stability Index (PSI)
- KL Divergence

**Application Metrics:**
- Throughput (Requests/minute)
- Error Rate
- Latency
- CPU usage

## Technology Stack

- **Database**: PostgreSQL (containerized)
- **Object Storage**: MinIO (for images)
- **ML Tracking**: MLflow
- **API**: FastAPI
- **Containerization**: Docker
- **Models**: Scikit-learn (XGBoost, Random Forest, Logistic Regression, SVM)
- **Testing**: Pytest
- **Monitoring**: Grafana Dashboard
- **Data Quality/Drift**: Custom monitoring pipeline

## Team Development

Each team member should:
1. Clone this repository
2. Set up local development environment (virtual environment + requirements)
3. Run Qi's setup scripts to initialize Docker infrastructure
4. Develop and test locally

## Contributing

1. Create feature branch from main
2. Make changes and test locally
3. Submit pull request
4. Ensure all tests pass
