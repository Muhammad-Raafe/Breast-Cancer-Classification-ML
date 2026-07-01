# 🎗️ Breast Cancer Diagnosis AI

An interactive machine learning web app that classifies breast tumors as **Benign** or **Malignant**, comparing three different classification algorithms — Decision Tree, Random Forest, and SVM — each hyperparameter-tuned with GridSearchCV. Built with Scikit-learn and deployed with Streamlit.

🔗 **Live Demo:** https://breast-cancer-classification-ml-raafe.streamlit.app/

⚠️ **Disclaimer:** This project is for educational and portfolio purposes only. It is not a medical diagnostic tool and should never be used for real clinical decisions.

---

## 🧠 Overview

This project uses the Wisconsin Breast Cancer Diagnostic dataset, which contains measurements computed from digitized images of fine needle aspirate (FNA) of breast masses. Three classifiers are trained, tuned, and compared side-by-side — letting the user pick which model powers the live prediction, and see exactly how each one performs.

---

## ✨ Features

- 🤖 **Multi-model comparison** — Decision Tree, Random Forest, and SVM trained and tuned in parallel, all visible in one dashboard
- 🔍 **Live diagnosis tool** — enter tumor measurements (or load a random real sample) and get an instant Benign/Malignant prediction with confidence score
- ⚙️ **Hyperparameter tuning transparency** — view the best parameters found via GridSearchCV for each model
- 📊 **Full evaluation suite** — accuracy, cross-validation score, confusion matrix, and classification report per model
- 📈 **Interactive Plotly visualizations** — model comparison bar chart, correlation heatmap, class distribution, and customizable feature scatter plots
- 🎨 **Premium dark-themed UI** with gradient accents and smooth interactions
- ⚡ **Cached model training** for fast app performance

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| ML Library | Scikit-learn |
| Models | Decision Tree, Random Forest, Support Vector Machine (SVM) |
| Hyperparameter Tuning | GridSearchCV (5-fold CV) |
| Preprocessing | StandardScaler, LabelEncoder |
| Visualization | Plotly, Seaborn, Matplotlib |
| Web Framework | Streamlit |
| Data Handling | Pandas, NumPy |

---

## 📂 Dataset

The model is trained on the **Wisconsin Breast Cancer Diagnostic Dataset**, containing 30 numeric features computed from digitized FNA images (radius, texture, perimeter, area, smoothness, etc.), each reported as mean, standard error, and "worst" (largest) value.

**Target:**
- `diagnosis` — M (Malignant) or B (Benign), label-encoded to 1/0

Dataset source: [UCI Machine Learning Repository / Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## ⚙️ How It Works

1. **Data Preprocessing** — Dropped irrelevant columns (`id`, `Unnamed: 32`), checked for nulls, label-encoded the diagnosis column
2. **Feature/Target Split** — Separated the 30 measurement features (`x`) from the diagnosis label (`y`)
3. **Train/Test Split** — 80/20 split for training and evaluation
4. **Model 1 — Decision Tree** — Tuned `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` via GridSearchCV
5. **Model 2 — Random Forest** — Tuned `n_estimators`, `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf` via GridSearchCV
6. **Model 3 — SVM** — Features scaled with `StandardScaler` (SVM is distance-based), tuned `C`, `kernel`, `gamma` via GridSearchCV
7. **Evaluation** — Compared all three on accuracy, cross-validation score, confusion matrix, and classification report
8. **Deployment** — Wrapped in an interactive Streamlit app where the user can pick any of the three models for live predictions

---

## 🚀 Running Locally

1. Clone the repository
```bash
git clone https://github.com/Muhammad-Raafe/Breast-Cancer-Classification-ML.git
cd Breast-Cancer-Classification-ML
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run app.py
```

Make sure `data.csv` is present in the project root directory.

---

## 📁 Project Structure

```
Breast-Cancer-Classification-ML/
│
├── app.py                # Streamlit web app (multi-model comparison + live prediction)
├── model_training.py      # Original model training & comparison script
├── data.csv                # Dataset
├── requirements.txt         # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Model Performance

All three models are evaluated on the same held-out test set, with accuracy, cross-validation score, confusion matrix, and full classification report displayed live in the app — letting you directly compare Decision Tree vs Random Forest vs SVM performance on this dataset.

---

## 🔮 Future Improvements

- Add feature importance visualization for tree-based models
- Add ROC curve and AUC score comparison across all three models
- Support batch CSV upload for bulk diagnosis
- Add SHAP-based model explainability for individual predictions

---

## 👤 Author

**Muhammad Raafe**
AI/ML Enthusiast | Building a portfolio in Machine Learning & Data Science

GitHub: [@Muhammad-Raafe](https://github.com/Muhammad-Raafe)
