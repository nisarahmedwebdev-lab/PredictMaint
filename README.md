# 🏭 PredictMaint — AI-Powered Predictive Maintenance Simulation

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Stable-success.svg)]()

## 📋 Table of Contents
- [Overview](#-overview)
- [Features](#-features)
- [Problem Statement](#-problem-statement)
- [Solution Architecture](#-solution-architecture)
- [Technology Stack](#-technology-stack)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [How It Works](#-how-it-works)
- [Dataset Format](#-dataset-format)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Overview

**PredictMaint** is an AI-powered predictive maintenance simulation system designed for factory machines and equipment. It simulates a fleet of machines streaming sensor data (temperature, vibration, pressure, current draw, RPM, and runtime hours), applies a machine learning model to predict impending failures and Remaining Useful Life (RUL), and presents everything through an animated, interactive dashboard.

**This is a software simulation** — no physical sensors or hardware required. Perfect for:
- 🎓 **Students** learning about AI in industrial applications
- 👨‍🔧 **Maintenance Engineers** exploring predictive maintenance concepts
- 🏗️ **Manufacturing Professionals** understanding AI-driven maintenance
- 📊 **Data Scientists** building industrial AI applications

---

## ✨ Features

### 🔄 Real-Time Simulation
- **Live Data Streaming**: Animated sensor data playback
- **Multiple Machines**: Monitor 5-10 virtual machines simultaneously
- **Speed Control**: Adjust simulation speed (1x to 10x)
- **Interactive Controls**: Play, Pause, Reset simulation

### 🤖 AI-Powered Predictions
- **Health Classification**: Normal / Warning / Critical status
- **RUL Estimation**: Remaining Useful Life in hours
- **Failure Probability**: Percentage likelihood of failure
- **ML vs Rule-Based**: Compare AI performance with traditional thresholds

### 📊 Interactive Dashboard
- **Live Trend Charts**: Real-time sensor data visualization
- **Health Gauges**: Color-coded health indicators (Green/Yellow/Red)
- **Fleet View**: All machines displayed with risk sorting
- **Feature Importance**: Understand what drives predictions

### 💡 Explainability
- **Feature Importance Charts**: Visualize top contributing factors
- **Natural Language Explanations**: Plain English reason for predictions
- **Model Transparency**: See which sensors influence decisions

### 📈 Model Evaluation
- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Regression Metrics**: MAE, RMSE for RUL predictions
- **Confusion Matrix**: Visual model performance
- **ML vs Rule-Based Comparison**: Side-by-side performance analysis

---

## 🎯 Problem Statement

### The Challenge
Unplanned equipment downtime is one of the largest hidden costs in manufacturing, costing industries billions annually. Traditional maintenance approaches include:

| Approach | Description | Issues |
|----------|-------------|--------|
| **Reactive** | Fix after breakdown | • Expensive emergency repairs<br>• Unplanned downtime<br>• Safety risks |
| **Preventive** | Fixed maintenance schedules | • Wasted resources<br>• Unnecessary repairs<br>• Still misses failures |

### The Solution
**Predictive Maintenance** uses live sensor data + machine learning to:
- 📊 **Monitor** machine health in real-time
- ⚠️ **Flag** problems before they cause failure
- 💰 **Reduce** downtime and maintenance costs
- 🔧 **Optimize** maintenance schedules
- 🏭 **Improve** safety and reliability

---

## 🏗️ Solution Architecture

<img width="1536" height="1024" alt="ChatGPT Image Jul 18, 2026, 11_09_34 AM" src="https://github.com/user-attachments/assets/565d0ee6-1eec-473e-90d7-76354fcfcea4" />



---

## 💻 Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **UI Framework** | Streamlit | Web-based interactive dashboard |
| **Visualization** | Plotly + Matplotlib | Real-time charts and animations |
| **Machine Learning** | scikit-learn | Random Forest models |
| **Data Processing** | Pandas + NumPy | Data manipulation and analysis |
| **Explainability** | Feature Importance | Model interpretability |
| **Data Generation** | Custom Python | Synthetic sensor simulation |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/PredictMaint.git
cd PredictMaint

pip install -r requirements.txt

# Solution 1: Use python -m
python -m streamlit run app.py

# Solution 2: Reinstall streamlit
pip install --upgrade streamlit<img width="1536" height="1024" alt="ChatGPT Image Jul 18, 2026, 11_09_34 AM" src="https://github.com/user-attachments/assets/328ed249-c04f-4f33-b90e-22c4a65d8ed9" />
