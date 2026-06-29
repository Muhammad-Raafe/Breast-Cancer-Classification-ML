# 🩺 Breast Cancer Prediction using Random Forest Classifier

## 📌 Project Overview

This project focuses on predicting whether a breast tumor is **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using the **Random Forest Classifier**. The complete machine learning workflow was implemented, including data preprocessing, model training, hyperparameter tuning, evaluation, and feature importance analysis.

---

## 🎯 Objectives

* Predict breast cancer diagnosis using machine learning.
* Perform data preprocessing and feature selection.
* Train a Random Forest Classification model.
* Improve model performance using hyperparameter tuning.
* Evaluate the model using multiple classification metrics.
* Identify the most influential features affecting predictions.

---

## 📂 Dataset

The project uses the **Breast Cancer Wisconsin Diagnostic Dataset**, containing measurements extracted from digitized images of breast cell nuclei.

**Target Variable:**

* **0 → Benign**
* **1 → Malignant**

---

## 🛠 Data Preprocessing

* Removed unnecessary `id` column.
* Encoded the target variable (`diagnosis`) into numerical values.
* Checked for missing values and duplicates.
* Split the dataset into training and testing sets.

---

## 🤖 Machine Learning Model

**Algorithm Used:**

* Random Forest Classifier

**Hyperparameters Tuned:**

* `criterion`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `random_state`

---

## 📊 Model Evaluation

The model was evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

  * Precision
  * Recall
  * F1-Score

**Final Accuracy:** **96%**

---

## 📈 Feature Importance

Random Forest was used to identify the most important features contributing to breast cancer prediction. A **Seaborn Feature Importance Bar Chart** was generated to visualize the top predictive features.

---

## 💻 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

## 🚀 Key Learning Outcomes

* Data preprocessing for medical datasets.
* Binary classification using Random Forest.
* Hyperparameter tuning for performance improvement.
* Model evaluation using classification metrics.
* Understanding and visualizing feature importance.
