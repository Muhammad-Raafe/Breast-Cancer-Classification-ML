import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: white; }
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        border: 1px solid #e91e8c;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #2d2d44;
        text-align: center;
        margin: 5px;
    }
    .section-header {
        color: #e91e8c;
        font-size: 18px;
        font-weight: bold;
        border-bottom: 2px solid #e91e8c;
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #e91e8c, #c2185b);
        color: white;
        border: none;
        padding: 15px;
        border-radius: 10px;
        font-size: 18px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #c2185b, #ad1457);
    }
    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #a0a0b0 !important;
        font-size: 13px;
    }
    .result-high {
        background: linear-gradient(135deg, #2d1b1b, #3d1a1a);
        border: 2px solid #e74c3c;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
    .result-low {
        background: linear-gradient(135deg, #1b2d1b, #1a3d1a);
        border: 2px solid #2ecc71;
        border-radius: 15px;
        padding: 25px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    df = pd.read_csv("breast-cancer.csv")
    le = LabelEncoder()
    df["diagnosis"] = le.fit_transform(df["diagnosis"])
    x = df.drop(["id", "diagnosis"], axis=1)
    y = df["diagnosis"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    model = RandomForestClassifier(
        random_state=42, criterion="gini",
        max_depth=5, min_samples_split=2,
        min_samples_leaf=2, n_estimators=100
    )
    model.fit(x_train, y_train)
    acc = accuracy_score(y_test, model.predict(x_test))
    importance_df = pd.DataFrame({
        "Feature": x.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    return model, x.columns.tolist(), acc, importance_df


model, feature_cols, acc, importance_df = train_model()

# --- Header ---
st.markdown(f"""
<div class="main-header">
    <h1>🎗️ Breast Cancer Risk Predictor</h1>
    <p style="color:#a0a0b0;">AI-powered tumor classification using Random Forest — Malignant vs Benign</p>
</div>
""", unsafe_allow_html=True)

# --- Top Metrics ---
m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#e91e8c;">{acc:.0%}</h2>
        <p style="color:#a0a0b0;">Model Accuracy</p>
    </div>""", unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#3498db;">569</h2>
        <p style="color:#a0a0b0;">Patients Trained On</p>
    </div>""", unsafe_allow_html=True)
with m3:
    st.markdown(f"""
    <div class="metric-card">
        <h2 style="color:#2ecc71;">Random Forest</h2>
        <p style="color:#a0a0b0;">ML Algorithm</p>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- Tabs ---
tab1, tab2 = st.tabs(["🔍 Predict", "📊 Feature Importance"])

with tab1:
    st.markdown('<p class="section-header">🧾 Tumor Measurements</p>', unsafe_allow_html=True)

    defaults = {
        "radius_mean": 14.0, "texture_mean": 19.0, "perimeter_mean": 92.0,
        "area_mean": 654.0, "smoothness_mean": 0.096, "compactness_mean": 0.104,
        "concavity_mean": 0.089, "concave points_mean": 0.049, "symmetry_mean": 0.181,
        "fractal_dimension_mean": 0.063, "radius_se": 0.405, "texture_se": 1.22,
        "perimeter_se": 2.87, "area_se": 40.3, "smoothness_se": 0.007,
        "compactness_se": 0.025, "concavity_se": 0.032, "concave points_se": 0.012,
        "symmetry_se": 0.021, "fractal_dimension_se": 0.004, "radius_worst": 16.3,
        "texture_worst": 25.7, "perimeter_worst": 107.0, "area_worst": 880.0,
        "smoothness_worst": 0.132, "compactness_worst": 0.254, "concavity_worst": 0.272,
        "concave points_worst": 0.115, "symmetry_worst": 0.290, "fractal_dimension_worst": 0.084
    }

    st.markdown("##### 📐 Mean Values")
    c1, c2, c3 = st.columns(3)
    mean_features = [f for f in feature_cols if "_mean" in f]
    inputs = {}

    for i, feat in enumerate(mean_features):
        col = [c1, c2, c3][i % 3]
        with col:
            inputs[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.4f")

    st.markdown("##### 📏 Standard Error Values")
    c4, c5, c6 = st.columns(3)
    se_features = [f for f in feature_cols if "_se" in f]

    for i, feat in enumerate(se_features):
        col = [c4, c5, c6][i % 3]
        with col:
            inputs[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.4f")

    st.markdown("##### 📊 Worst Values")
    c7, c8, c9 = st.columns(3)
    worst_features = [f for f in feature_cols if "_worst" in f]

    for i, feat in enumerate(worst_features):
        col = [c7, c8, c9][i % 3]
        with col:
            inputs[feat] = st.number_input(feat, value=float(defaults.get(feat, 0.0)), format="%.4f")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔬 Analyze Tumor"):
        input_df = pd.DataFrame([inputs])
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">📊 Analysis Results</p>', unsafe_allow_html=True)

        res_col, gauge_col = st.columns([1, 1])

        with res_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-high">
                    <h1>⚠️</h1>
                    <h2 style="color:#e74c3c;">Malignant Detected</h2>
                    <h3 style="color:white;">{probability:.0%} probability</h3>
                    <p style="color:#a0a0b0;">Please consult an oncologist immediately.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-low">
                    <h1>✅</h1>
                    <h2 style="color:#2ecc71;">Benign Tumor</h2>
                    <h3 style="color:white;">{probability:.0%} malignant probability</h3>
                    <p style="color:#a0a0b0;">Low risk detected. Regular checkups recommended.</p>
                </div>""", unsafe_allow_html=True)

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={"text": "Malignancy Risk Score", "font": {"color": "white", "size": 16}},
                number={"suffix": "%", "font": {"color": "white", "size": 40}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "white"},
                    "bar": {"color": "#e74c3c" if probability > 0.5 else "#2ecc71"},
                    "bgcolor": "#1a1a2e",
                    "steps": [
                        {"range": [0, 40], "color": "#1a3d1a"},
                        {"range": [40, 70], "color": "#3d3d1a"},
                        {"range": [70, 100], "color": "#3d1a1a"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.75,
                        "value": 50
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor="#0e1117",
                font={"color": "white"},
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown('<p class="section-header">📊 Top 10 Most Important Features</p>', unsafe_allow_html=True)
    fig2 = px.bar(
        importance_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="RdPu",
    )
    fig2.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font={"color": "white"},
        yaxis={"autorange": "reversed"},
        height=400,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)
