# 🧬 Breast Cancer Prediction App

This is a **Machine Learning web application** built with Streamlit that predicts whether a breast tumor is **benign or malignant** based on cellular characteristics.

---

## 📊 Project Overview

The model is trained using the Breast Cancer Wisconsin dataset from scikit-learn.  
It uses a **Logistic Regression classifier** trained on selected morphological features of tumor cells.

The application allows users to interactively explore how different features influence the prediction.

---

## ⚙️ Features Used

The model uses the following features:

- Worst Radius  
- Worst Perimeter  
- Mean Perimeter  
- Worst Concave Points  
- Mean Concave Points  

---

## 🧠 Machine Learning Model

- Algorithm: Logistic Regression  
- Library: scikit-learn  
- Type: Binary classification  
- Output: Probability of benign vs malignant tumor  

---

## 📈 App Features

- Interactive sliders to adjust patient parameters  
- Real-time prediction of tumor type  
- Probability visualization  
- Feature contribution analysis  
- Educational interpretation of results  

---

## ⚠️ Disclaimer

This project is for **educational purposes only**.  
It is NOT a medical diagnostic tool and should not be used in clinical decisions.

---

## 🚀 How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
