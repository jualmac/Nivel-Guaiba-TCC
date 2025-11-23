# Nivel_TCC
Projeto baseado em predições do nível da água locais baseado em análise e predição de Séries Temporais.

Projeto realizado para como trabalho de conclusão de curso para o curso de Bacharelado em Física pela Universidade Federal do Rio Grande do Sul. 

# Docker Setup
This project uses Docker to containerize each module for better separation of concerns and easier deployment.

## Architecture

The project is divided into 4 containerized services:

1. **API Service** (`Dockerfile.api`) - Collects data from HidroWeb API
2. **Database Service** (`Dockerfile.database`) - Runs ETL pipeline (cleaning, transformation, imputation)
3. **Backend Service** (`Dockerfile.backend`) - Trains machine learning models
4. **Frontend Service** (`Dockerfile.frontend`) - Runs Streamlit dashboard

All services share the same database file via Docker volumes.

## Prerequisites

- Docker (20.10+)
- Docker Compose (2.0+)

## Quick Start

### Build all services
```bash
docker-compose build
```

### Run the complete pipeline
```bash
# Run API data collection;
docker-compose up api

# Run ETL pipeline;
docker-compose up database

# Run model training;
docker-compose up backend

# Run frontend dashboard;
docker-compose up frontend
```

### Run all services sequentially
```bash
docker-compose up
```

The frontend dashboard will be available at: http://localhost:8501

## Individual Service Commands

### API - Data Collection
```bash
docker-compose up api
```
Collects raw data from HidroWeb API and stores in DuckDB.

### Database - ETL Pipeline
```bash
docker-compose up database
```
Processes raw data: cleaning, gap filling, outlier removal, imputation, and melting.

### Backend - Model Training
```bash
docker-compose up backend
```
Trains ML models (SARIMA, LSTM, XGBoost, LightGBM) on processed data.

### Frontend - Dashboard
```bash
docker-compose up frontend
```
Launches interactive Streamlit dashboard on port 8501.

## Volumes

The docker-compose setup creates two shared volumes:

- `./data` - Database files (DuckDB)
- `./models` - Trained model artifacts

## Development Mode

To rebuild after code changes:
```bash
docker-compose build --no-cache <service_name>
docker-compose up <service_name>
```

## Environment Variables

Each service respects environment variables defined in `docker-compose.yml`:
- `PYTHONUNBUFFERED=1` - Forces stdout/stderr streams to be unbuffered

## Cleanup

Remove all containers and images:
```bash
docker-compose down
docker-compose down --rmi all  # Also remove images
```

Remove volumes (WARNING: deletes database and models):
```bash
docker-compose down -v
```

## Notes

- API, Database, and Backend services run once and stop (`restart: "no"`)
- Frontend service runs continuously (`restart: unless-stopped`)
- All services use Python 3.11-slim base image
- Dependencies are installed from `requirements.txt`