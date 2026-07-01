import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Diagnosis AI",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- VIP DARK THEME CSS ----------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #171420 0%, #0a0910 100%);
        color: #f5f6fa;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #ff6fb5, #ff9ecb, #c77dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        letter-spacing: 1px;
    }

    .sub-title {
        text-align: center;
        color: #9a92a8;
        font-size: 1.05rem;
        margin-bottom: 30px;
        font-weight: 400;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #1b1626, #221b2e);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #2c2438;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }

    div[data-testid="stMetric"] label {
        color: #9a92a8 !important;
    }

    .result-box {
        padding: 28px;
        border-radius: 18px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 800;
        margin-top: 18px;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.4);
    }

    .benign-box {
        background: linear-gradient(145deg, #003d2b, #002417);
        border: 2px solid #00f5a0;
        color: #00f5a0;
    }

    .malignant-box {
        background: linear-gradient(145deg, #3d0014, #240009);
        border: 2px solid #ff4d6d;
        color: #ff4d6d;
    }

    .stButton>button {
        background: linear-gradient(90deg, #c77dff, #ff6fb5);
        color: white;
        font-weight: 700;
        border-radius: 12px;
        border: none;
        padding: 12px 0px;
        font-size: 1rem;
        transition: 0.3s;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 20px rgba(199, 125, 255, 0.5);
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #130f1b, #0a0910);
        border-right: 1px solid #251f30;
    }

    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #ff6fb5;
        margin-top: 10px;
        margin-bottom: 10px;
        border-left: 4px solid #c77dff;
        padding-left: 10px;
    }

    hr {
        border-color: #251f30;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------- LOAD + TRAIN (cached) ----------------
@st.cache_resource
def load_and_train():
    df = pd.read_csv("data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")

    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])  # M=1, B=0 typically

    x = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # ---- Decision Tree ----
    dc_grid = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        cv=5, scoring="accuracy"
    )
    dc_grid.fit(x_train, y_train)
    dc_pred = dc_grid.predict(x_test)

    # ---- Random Forest ----
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        {
            "n_estimators": [100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2]
        },
        cv=5, scoring="accuracy"
    )
    rf_grid.fit(x_train, y_train)
    rf_pred = rf_grid.predict(x_test)

    # ---- SVM (needs scaling) ----
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    sv_grid = GridSearchCV(
        SVC(probability=True, random_state=42),
        {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"],
            "gamma": ["scale", "auto"]
        },
        cv=5, scoring="accuracy"
    )
    sv_grid.fit(x_train_scaled, y_train)
    sv_pred = sv_grid.predict(x_test_scaled)

    results = {
        "Decision Tree": {
            "model": dc_grid, "pred": dc_pred, "needs_scaling": False,
            "accuracy": accuracy_score(y_test, dc_pred),
            "cm": confusion_matrix(y_test, dc_pred),
            "report": classification_report(y_test, dc_pred, output_dict=True),
            "best_params": dc_grid.best_params_,
            "cv_score": dc_grid.best_score_
        },
        "Random Forest": {
            "model": rf_grid, "pred": rf_pred, "needs_scaling": False,
            "accuracy": accuracy_score(y_test, rf_pred),
            "cm": confusion_matrix(y_test, rf_pred),
            "report": classification_report(y_test, rf_pred, output_dict=True),
            "best_params": rf_grid.best_params_,
            "cv_score": rf_grid.best_score_
        },
        "Support Vector Machine": {
            "model": sv_grid, "pred": sv_pred, "needs_scaling": True,
            "accuracy": accuracy_score(y_test, sv_pred),
            "cm": confusion_matrix(y_test, sv_pred),
            "report": classification_report(y_test, sv_pred, output_dict=True),
            "best_params": sv_grid.best_params_,
            "cv_score": sv_grid.best_score_
        }
    }

    return df, x, y, scaler, results, x.columns.tolist(), le

df, x_full, y_full, scaler, results, feature_names, le = load_and_train()

# ---------------- HEADER ----------------
st.markdown('<p class="main-title">🎗️ Breast Cancer Diagnosis AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Multi-Model Comparison: Decision Tree vs Random Forest vs SVM</p>', unsafe_allow_html=True)

# ---------------- SIDEBAR: MODEL SELECTOR + STATS ----------------
with st.sidebar:
    st.markdown('<p class="section-header">🤖 Choose Model</p>', unsafe_allow_html=True)
    selected_model_name = st.selectbox("Active model for prediction:", list(results.keys()), index=1)
    active = results[selected_model_name]

    st.divider()
    st.markdown('<p class="section-header">⚙️ Model Performance</p>', unsafe_allow_html=True)
    st.metric("Test Accuracy", f"{active['accuracy']*100:.2f}%")
    st.metric("Cross-Val Score", f"{active['cv_score']*100:.2f}%")
    with st.expander("Best Hyperparameters"):
        st.json(active["best_params"])

    st.divider()
    st.markdown('<p class="section-header">📋 Dataset Snapshot</p>', unsafe_allow_html=True)
    st.metric("Total Samples", len(df))
    st.metric("Benign Cases", int((df["diagnosis"] == 0).sum()))
    st.metric("Malignant Cases", int((df["diagnosis"] == 1).sum()))

# ---------------- MODEL COMPARISON ----------------
st.markdown('<p class="section-header">📊 Model Comparison</p>', unsafe_allow_html=True)

comp_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [results[m]["accuracy"] * 100 for m in results],
    "CV Score": [results[m]["cv_score"] * 100 for m in results]
})

c1, c2 = st.columns([1.2, 1])
with c1:
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["Accuracy"], name="Test Accuracy", marker_color="#ff6fb5"))
    fig_comp.add_trace(go.Bar(x=comp_df["Model"], y=comp_df["CV Score"], name="CV Score", marker_color="#c77dff"))
    fig_comp.update_layout(
        template="plotly_dark", barmode="group", height=380,
        title="Accuracy vs Cross-Validation Score",
        yaxis_title="Score (%)",
        plot_bgcolor="#0a0910", paper_bgcolor="#0a0910",
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with c2:
    st.dataframe(comp_df.round(2), use_container_width=True, hide_index=True)
    best_model = comp_df.loc[comp_df["Accuracy"].idxmax(), "Model"]
    st.success(f"🏆 Best performing model: **{best_model}**")

# ---------------- LIVE PREDICTION ----------------
st.divider()
st.markdown('<p class="section-header">🔍 Diagnose a New Sample</p>', unsafe_allow_html=True)
st.caption(f"Using **{selected_model_name}** — enter cell nuclei measurements below, or load a random sample from the dataset.")

if "sample_values" not in st.session_state:
    st.session_state.sample_values = x_full.iloc[0].to_dict()

load_random = st.button("🎲 Load Random Sample from Dataset")
if load_random:
    st.session_state.sample_values = x_full.sample(1).iloc[0].to_dict()

mean_cols = [c for c in feature_names if c.endswith("_mean")]
se_cols = [c for c in feature_names if c.endswith("_se")]
worst_cols = [c for c in feature_names if c.endswith("_worst")]

input_vals = {}
tab_mean, tab_se, tab_worst = st.tabs(["📏 Mean Values", "📐 Standard Error", "⚠️ Worst Values"])

def render_inputs(cols, tab):
    with tab:
        grid_cols = st.columns(3)
        for i, col_name in enumerate(cols):
            with grid_cols[i % 3]:
                input_vals[col_name] = st.number_input(
                    col_name.replace("_", " ").title(),
                    value=float(st.session_state.sample_values[col_name]),
                    format="%.4f",
                    key=f"input_{col_name}"
                )

render_inputs(mean_cols, tab_mean)
render_inputs(se_cols, tab_se)
render_inputs(worst_cols, tab_worst)

predict_btn = st.button("🚀 Run Diagnosis", use_container_width=True, type="primary")

if predict_btn:
    input_df = pd.DataFrame([input_vals])[feature_names]

    if active["needs_scaling"]:
        input_processed = scaler.transform(input_df)
    else:
        input_processed = input_df

    pred = active["model"].predict(input_processed)[0]
    proba = active["model"].predict_proba(input_processed)[0]

    benign_prob = proba[0] * 100
    malignant_prob = proba[1] * 100

    if pred == 0:
        st.markdown(f'<div class="result-box benign-box">✅ BENIGN ({benign_prob:.1f}% confidence)</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-box malignant-box">🚨 MALIGNANT ({malignant_prob:.1f}% confidence)</div>', unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=[benign_prob, malignant_prob],
        y=["Benign", "Malignant"],
        orientation="h",
        marker_color=["#00f5a0", "#ff4d6d"],
        text=[f"{benign_prob:.1f}%", f"{malignant_prob:.1f}%"],
        textposition="auto"
    ))
    fig.update_layout(
        template="plotly_dark",
        title="Prediction Confidence",
        xaxis_title="Probability (%)",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="#0a0910",
        paper_bgcolor="#0a0910"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption("⚠️ This tool is for educational/portfolio purposes only and is not a medical diagnostic device.")

# ---------------- CONFUSION MATRIX + CLASSIFICATION REPORT ----------------
st.divider()
st.markdown('<p class="section-header">📈 Detailed Evaluation</p>', unsafe_allow_html=True)

eval_col1, eval_col2 = st.columns(2)

with eval_col1:
    st.write(f"**Confusion Matrix — {selected_model_name}**")
    fig_cm = px.imshow(
        active["cm"], text_auto=True, color_continuous_scale="Magenta",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Benign", "Malignant"], y=["Benign", "Malignant"]
    )
    fig_cm.update_layout(
        template="plotly_dark", height=350,
        plot_bgcolor="#0a0910", paper_bgcolor="#0a0910",
        margin=dict(l=10, r=10, t=20, b=10)
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with eval_col2:
    st.write(f"**Classification Report — {selected_model_name}**")
    report_df = pd.DataFrame(active["report"]).transpose().round(3)
    st.dataframe(report_df, use_container_width=True)

# ---------------- EXPLORATORY VISUALS ----------------
st.divider()
st.markdown('<p class="section-header">🔬 Data Exploration</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔗 Feature Correlation", "📊 Class Distribution", "🌌 Feature Scatter"])

with tab1:
    top_features = mean_cols
    corr = df[top_features + ["diagnosis"]].corr()
    fig_corr = px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r", aspect="auto"
    )
    fig_corr.update_layout(
        template="plotly_dark", height=500,
        plot_bgcolor="#0a0910", paper_bgcolor="#0a0910"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

with tab2:
    dist_df = df["diagnosis"].map({0: "Benign", 1: "Malignant"}).value_counts().reset_index()
    dist_df.columns = ["Class", "Count"]
    fig_dist = px.pie(
        dist_df, names="Class", values="Count", hole=0.45,
        color="Class",
        color_discrete_map={"Benign": "#00f5a0", "Malignant": "#ff4d6d"}
    )
    fig_dist.update_layout(
        template="plotly_dark", height=400,
        plot_bgcolor="#0a0910", paper_bgcolor="#0a0910"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with tab3:
    feature_x = st.selectbox("X-axis feature", mean_cols, index=0)
    feature_y = st.selectbox("Y-axis feature", mean_cols, index=1)
    plot_df = df.copy()
    plot_df["diagnosis"] = plot_df["diagnosis"].map({0: "Benign", 1: "Malignant"})
    fig_scatter = px.scatter(
        plot_df, x=feature_x, y=feature_y, color="diagnosis",
        color_discrete_map={"Benign": "#00f5a0", "Malignant": "#ff4d6d"},
        opacity=0.75
    )
    fig_scatter.update_layout(
        template="plotly_dark", height=450,
        plot_bgcolor="#0a0910", paper_bgcolor="#0a0910"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- RAW DATA ----------------
with st.expander("📋 View Raw Dataset Sample"):
    st.dataframe(df.sample(10, random_state=1), use_container_width=True)
