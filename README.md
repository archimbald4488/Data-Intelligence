# Student Performance Prediction

This project is part of the **Data Intelligence** course deliverables. The goal is to **predict student dropout risk** using the [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance).

---

## Project Overview
- **Objective:** Predict dropout risk (binary classification) from student demographic, behavioral, and academic features.  
- **Dataset:** Student Performance dataset (Math & Portuguese courses).  
- **Target:** Dropout = 1 if `G3 < 10`, else 0.  
- **Methods:**  
  - Logistic Regression (baseline, interpretable)  
  - Random Forest Classifier  
  - Gradient Boosting Classifier  

---

## 🔄 Data Intelligence Pipeline
1. **Data Ingestion**: Load CSV data (`student-mat.csv` or `student-por.csv`).  
2. **Preprocessing**: Encode categorical features, scale numerical ones, and create dropout label.  
3. **Feature Engineering**: Exclude `G3` to avoid leakage; retain `G1`, `G2`, and other features.  
4. **Model Training**: Train Logistic Regression, Random Forest, and Gradient Boosting.  
5. **Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrices.  
6. **Visualization**: Feature importance plots, ROC curves, confusion matrices.  

---

## Results Summary
- **Logistic Regression:** 94% accuracy, ROC AUC = 0.97 (best model).  
- **Random Forest:** 90% accuracy, ROC AUC = 0.97.  
- **Gradient Boosting:** 89% accuracy, ROC AUC = 0.97.  
- Key predictors: early grades (`G1`, `G2`), absences, and study time.  

---

## Database link:
https://archive.ics.uci.edu/dataset/320/student+performance (file needed is named student.mat-csv, extract from downloaded zip file)

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/<your-username>/student-performance-prediction.git
cd student-performance-prediction
pip install -r requirements.txt
