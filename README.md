# 📈 Store Sales Forecasting System

End-to-end **Store Sales Forecasting** project built with **Machine Learning + FastAPI + Streamlit**.  
The system predicts daily store sales using historical patterns, promotions, holidays and rolling statistics.

This project demonstrates a **production-style ML workflow**:  
data analysis → feature engineering → model training → API serving → interactive dashboard.

---

## 🚀 Project Overview

- **Problem**: Forecast daily store sales accurately
- **Solution**: Train a regression model with time-series features and serve it via API
- **Model**: RandomForest Regressor
- **Deployment**: FastAPI (backend) + Streamlit (frontend)

---

## 🧠 Features Used

Time-series and business-driven features:

- `onpromotion` – Promotion count
- `is_holiday` – Holiday indicator (0/1)
- `lag_1` – Previous day sales
- `lag_7` – Sales from same day last week
- `rolling_7_mean` – 7-day rolling average
- `rolling_14_mean` – 14-day rolling average

---

## 📊 Model Performance

After training the model:

- **MAE**: 86.85  
- **RMSE**: 359.42  

Metrics are stored together with the model artifact for reproducibility.

---

## 🗂 Project Structure

store-sales-ai/
│
├── api/
│ ├── main.py # FastAPI backend
│ └── init.py
│
├── dashboard/
│ └── app.py # Streamlit dashboard
│
├── src/
│ └── train_model.py # Model training pipeline
│
├── artifacts/
│ └── model_bundle.joblib # Trained model + metadata
│
├── data/
│ ├── raw/
│ └── processed/
│
├── notebooks/
│ └── 01_eda.ipynb # Exploratory Data Analysis
│
├── requirements.txt
└── README.md

QUICKSTART:
python -m src.train_model
python -m uvicorn api.main:app --reload --port 8000
streamlit run dashboard/app.py
