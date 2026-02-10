# Employee Attrition Prediction Model

<div align="center">
  <img src="https://img.shields.io/badge/MLOps-Hub-blue?style=for-the-badge" alt="MLOps Hub" />
  <img src="https://img.shields.io/badge/python-3.12.x-blue?style=for-the-badge" alt="Python Version" />
  <img src="https://img.shields.io/badge/status-Active-green?style=for-the-badge" alt="Status" />
</div>

<hr />


## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Features](#features)
- [Project Structure](#project-structure)
  - [Directory Details](#directory-details)
- [Tech Stack](#tech-stack)
- [Implementation of ML Model](#implementation-of-ml-model)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Model Development](#2-model-development)
  - [3.a Testing](#3a-testing)
  - [3.b Frontend](#3b-frontend)
- [Setup & Installation](#setup--installation)
  - [Step 1: Create Virtual Environment](#step-1-create-virtual-environment)
  - [Step 2: Install Dependencies](#step-2-install-dependencies)
  - [Step 3: Run Data Pipeline](#step-3-run-data-pipeline)
  - [Step 4: Train & Evaluate Model](#step-4-train--evaluate-model)
  - [Step 5: Test Predictions](#step-5-test-predicitions)
  - [Step 6: Run Web Application](#step-6-run-web-application)
- [Key Findings](#key-findings)
- [Contributing](#contributing)
- [License](#license)


## Overview

This project is an **Employee Attrition Prediction System** built using machine learning. It predicts whether an employee will leave (attrition) or stay at a company based on various employee and workplace factors. The model uses classification algorithms to identify at-risk employees, helping organizations implement targeted retention strategies.

**Business Value:**
- Identify employees likely to leave the organization
- Enable proactive retention interventions
- Reduce turnover costs and improve workforce planning
- Support HR decision-making with data-driven insights


## Datasets

The main dataset used is [employee_attrition.csv](./datasets/employee_attrition.csv), containing 74,500 employee records with both training and test data.

**Dataset Features:**

| Feature | Description |
|---------|-------------|
| Employee ID | Unique identifier for each employee |
| Age | Age of the employee |
| Gender | Gender (Male/Female) |
| Years at Company | Tenure in years |
| Job Role | Position/Department |
| Monthly Income | Salary in monthly terms |
| Work-Life Balance | Rating (Poor/Good/Excellent) |
| Job Satisfaction | Level (Low/Medium/High) |
| Performance Rating | Employee performance (Low/Average/High) |
| Number of Promotions | Career progression count |
| Overtime | Whether employee works overtime (Yes/No) |
| Distance from Home | Commute distance in km |
| Education Level | Highest education attained |
| Marital Status | Single/Married/Divorced |
| Number of Dependents | Family dependents count |
| Job Level | Position hierarchy level |
| Company Size | Organization size (Small/Medium/Large) |
| Company Tenure | Time at current company |
| Remote Work | Remote work eligibility (Yes/No) |
| Leadership Opportunities | Career growth potential |
| Innovation Opportunities | Innovation involvement |
| Company Reputation | Market reputation |
| Employee Recognition | Recognition programs |
| **Attrition** | **Target: Stayed/Left** |
| dataset_type | Train/Test split indicator |

## Features

**Key Factors Influencing Employee Attrition:**
- **Compensation**: Monthly income, job level
- **Work Environment**: Work-life balance, overtime, remote work, distance from home
- **Career Development**: Promotions, leadership opportunities, innovation participation
- **Job Satisfaction**: Overall satisfaction, performance ratings, recognition
- **Demographics**: Age, tenure, marital status, education level
- **Company Factors**: Company size, reputation, tenure at organization


## Project Structure

```
employee-attrition-model/
│
├── 📂 datasets/
│   ├── employee_attrition.csv           # Main dataset (74,500 employee records)
│   └── 📂 data-preparation/             # Processed datasets at each pipeline stage
├── 📂 src/
│   ├── 📂 config/                      # Configuration for project paths
│   │
│   ├── 📂 data_preparation/            # Data Engineering Pipeline (Step 1-6)
│   │   ├── 01_ingestion.py             
│   │   ├── 02_validation.py            
│   │   ├── 03_eda.py                   
│   │   ├── 04_cleaning.py             
│   │   ├── 05_feature_engg.py          
│   │   ├── 06_preprocessing.py         
│   │
│   ├── 📂 model_development/           # ML Model Development Pipeline (Step 7-10)
│   │   ├── 07_training.py              
│   │   ├── 08_evaluation.py            
│   │   ├── 09_cross_validation.py      
│   │   ├── 10_tuning.py                
│   │
│   └── 📂 testing/
│       └── predict.py                  # Inference module for making predictions on new data
│
├── 📂 frontend/                        # Flask application entry point
├── 📂 notebook/                        # Jupyter Notebook for experimentation and exploration
|                  
├── README.md                           # Project documentation (this file)
└── requirements.txt                    # Python package dependencies
```

### Directory Details

| Directory | Purpose |
|-----------|---------|
| `artifacts/` | Stores trained models, metrics, and performance results |
| `datasets/` | Contains raw and processed data at each pipeline stage |
| `src/config/` | Project configuration and path management |
| `src/data_preparation/` | Data engineering and preprocessing modules (6 sequential steps) |
| `src/model_development/` | ML model training, evaluation, and tuning (4 sequential steps) |
| `src/testing/` | Prediction/inference module for model testing |
| `frontend/` | Flask web application with UI for predictions |
| `notebook/` | Jupyter notebooks for data exploration and experimentation |


## Tech Stack

- **Python Version**: 3.12+
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
- **Data Validation**: pandera
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Flask, Flask-CORS
- **Model Format**: Pickle (.pkl)


## Implementation of ML Model

### 1. Data Preparation
- **Ingestion**: Load raw data from CSV
- **Validation**: Schema validation using Pandera
- **EDA**: Exploratory Data Analysis - understand data patterns
- **Cleaning**: Handle missing values, outliers, inconsistencies
- **Feature Engineering**: Create new features, encoding categorical variables
- **Preprocessing**: Scaling, normalization, train-test split

### 2. Model Development
- **Training**: Train classification models (Logistic Regression, Random Forest, etc.)
- **Evaluation**: Accuracy, precision, recall, F1-score, confusion matrix
- **Cross-Validation**: K-fold validation for robust assessment
- **Hyperparameter Tuning**: GridSearch/RandomSearch for optimal parameters

### 3.a. Testing
- **predict.py**: Test prediction of the trained model in this file.

### 3.b. Frontend
- Flask-based web application for model inference
- Real-time predictions on new employee data


## Setup & Installation

#### Step 1: Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Run Data Pipeline

Execute each data engineering step sequentially:

```bash
cd src

python -m data_preparation.01_ingestion
python -m data_preparation.02_validation
python -m data_preparation.03_eda
python -m data_preparation.04_cleaning
python -m data_preparation.05_feature_engg
python -m data_preparation.06_preprocessing

```

#### Step 4: Train & Evaluate Model

Once data is preprocessed, run model development pipeline:

```bash
cd src

python -m model_development.07_training
python -m model_development.08_evaluation
python -m model_development.09_cross_validation
python -m model_development.10_tuning

```

#### Step-5: Test Predicitions

```bash
cd src
python -m testing.predict
```

#### Step 6: Run Web Application

```bash
cd ../frontend
python app.py
```

Access the application at `http://localhost:5000`


## Key Findings

- Identify primary drivers of employee attrition
- Understand demographic and workplace patterns
- Data-driven recommendations for retention strategies


## Contributing

Please read our [Contributing Guidelines](CONTRIBUTION.md) before submitting pull requests.

Contributions are welcome! Please follow standard Git workflow:
1. Create a feature branch
2. Make your changes
3. Submit a pull request


## License
This project is under [MIT Licence](LICENCE) support.
