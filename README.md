# 🧠 NeuroMaintain
### An Explainable AI-Powered Predictive Maintenance System

**Predict. Detect. Explain. Prevent.**

NeuroMaintain is an intelligent predictive maintenance platform that leverages Machine Learning, Deep Learning, Explainable AI, and MLOps to predict industrial machine failures before they occur.

Built using the **AI4I 2020 Predictive Maintenance Dataset**, the system analyzes machine operating conditions and sensor parameters to estimate failure probability, detect abnormal machine behaviour, classify operational risks, and generate automated maintenance work orders.

## 🚀 Key Features

- **Machine Failure Prediction:** ML-based failure probability estimation using industrial sensor data.
- **Deep Learning Anomaly Detection:** PyTorch-based Autoencoder to identify unusual machine behaviour.
- **Intelligent Risk Classification:** Rule-based Triage Agent categorizes machine conditions into NORMAL, WARNING, and CRITICAL.
- **Explainable AI:** SHAP-based interpretation of important prediction features.
- **Automated Maintenance Work Orders:** Generates actionable maintenance recommendations based on machine risk.
- **Interactive Dashboard:** Streamlit interface for sensor inputs, failure predictions, anomaly alerts, and maintenance insights.
- **MLOps Monitoring:** Prediction logging, model health checks, and monitoring summaries.

## 🏗️ System Architecture

```text
       AI4I 2020 Dataset / Sensor Inputs
                     |
                     v
             Data Preprocessing
                     |
                     v
             Feature Engineering
                     |
             +-------+-------+
             |               |
             v               v
       XGBoost Model     Autoencoder
       Failure           Anomaly
       Prediction        Detection
             |               |
             +-------+-------+
                     |
                     v
              SHAP Explainability
                     |
                     v
              Rule-Based Triage
                   Agent
                     |
                     v
              Risk Classification
                     |
                     v
             Maintenance Work Order
                     |
                     v
             Streamlit Dashboard
                     |
                     v
              MLOps Monitoring
```

## 🛠️ Technology Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Explainable AI | SHAP |
| Visualization | Matplotlib, Seaborn |
| Dashboard | Streamlit |
| MLOps | Custom Prediction Logging & Model Monitoring |
| Dataset | AI4I 2020 Predictive Maintenance Dataset |
| Version Control | Git, GitHub |

## 📊 Dataset

**AI4I 2020 Predictive Maintenance Dataset**

Source: UCI Machine Learning Repository.

The dataset contains 10,000 synthetic industrial records representing machine operating conditions and failure events.

Important features include:

- Air Temperature
- Process Temperature
- Rotational Speed
- Torque
- Tool Wear
- Machine Type

The dataset also includes machine failure labels and different failure modes, making it suitable for predictive maintenance experimentation.

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/NeuroMaintain.git
cd NeuroMaintain
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Environment (Windows)

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Application

```bash
streamlit run dashboard/app.py
```

## 🔄 Workflow

1. Collect machine operating parameters.
2. Preprocess sensor data and engineer relevant features.
3. Predict failure probability using the trained ML model.
4. Detect abnormal patterns using the Autoencoder.
5. Interpret predictions using Explainable AI.
6. Evaluate machine risk through the Triage Agent.
7. Generate maintenance work orders.
8. Log predictions and monitor model health.
9. Display results through the Streamlit dashboard.

## 🚦 Risk Classification

| Risk Level | Interpretation |
|---|---|
| NORMAL | Machine operating under acceptable conditions |
| WARNING | Potential abnormal behaviour requiring attention |
| CRITICAL | High predicted failure risk requiring immediate inspection |

## 🔮 Future Scope

- Real-time industrial IoT integration.
- Advanced model drift detection.
- Automated model retraining pipelines.
- Integration with industrial maintenance management systems.
- Cloud-based deployment and scalable monitoring.

## 👨‍💻 Project Information

**Project Name:** NeuroMaintain  
**Domain:** AI | Machine Learning | Deep Learning | Predictive Maintenance | MLOps  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**Interface:** Streamlit

---

⭐ If you find NeuroMaintain interesting, consider starring the repository!

