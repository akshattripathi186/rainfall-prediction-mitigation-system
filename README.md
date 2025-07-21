# Rainfall Prediction and Mitigation System 🌧️

A machine learning-based application to predict rainfall using real-time weather data, visualize forecasts, and support proactive mitigation planning.

## 🔍 Overview
This project uses the **Open-Meteo API**, **FastAPI**, and **Matplotlib** to provide:
- Rainfall prediction with over **80% accuracy**
- Visualized trends of rainfall forecasts
- Cloud deployment using **AWS EC2**
- Handles over **500+ daily requests**

## 🚀 Tech Stack
- **Frontend**: HTML, CSS (Basic UI for results)
- **Backend**: Python, FastAPI
- **Machine Learning**: Sklearn
- **APIs**: Open-Meteo Weather API
- **Visualization**: Matplotlib
- **Deployment**: AWS EC2

## 🧠 Features
- Fetches real-time weather data
- Predicts rainfall using trained ML model
- Provides actionable insights and charts
- Deployable as a REST API

## 🛠️ Setup Instructions

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/rainfall-prediction-mitigation-system.git
   cd rainfall-prediction-mitigation-system
2. Create virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies
   pip install -r requirements.txt

4.Run the FastAPI server
  uvicorn main:app --reload
  
