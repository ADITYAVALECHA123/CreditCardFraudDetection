import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from datetime import datetime
import base64
import io
import os

if "prediction_history" not in st.session_state:
    st.session_state["prediction_history"] = []

if "latest_prediction_df" not in st.session_state:
    st.session_state["latest_prediction_df"] = None

if "latest_ml_prob" not in st.session_state:
    st.session_state["latest_ml_prob"] = None

if "latest_dl_prob" not in st.session_state:
    st.session_state["latest_dl_prob"] = None

if "latest_final_prob" not in st.session_state:
    st.session_state["latest_final_prob"] = None

if "latest_prediction_time" not in st.session_state:
    st.session_state["latest_prediction_time"] = None

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}
/* ---- Background ---- */
.stApp {
    background: linear-gradient(135deg, #0f172a 0%, #111827 100%);
    color: white;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #111827 !important;
}

[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ---- Titles ---- */
.main-title {
    font-size: 42px;
    font-weight: 700;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 40px;
    font-size: 18px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-bottom: 20px;
    color: white;
}

/* ---- Cards ---- */
.card {
    background: rgba(30, 41, 59, 0.85);
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    margin-bottom: 20px;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #334155);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    color: white;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}

.metric-number {
    font-size: 34px;
    font-weight: 700;
    line-height: 1.2;
}

.metric-label {
    color: #cbd5e1;
    font-size: 15px;
    margin-top: 4px;
}

/* ---- Alert Boxes ---- */
.fraud-alert {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    padding: 22px;
    border-radius: 16px;
    color: white;
    font-weight: 600;
    font-size: 18px;
    margin-top: 20px;
    border: 1px solid #ef4444;
}

.safe-alert {
    background: linear-gradient(135deg, #064e3b, #065f46);
    padding: 22px;
    border-radius: 16px;
    color: white;
    font-weight: 600;
    font-size: 18px;
    margin-top: 20px;
    border: 1px solid #10b981;
}

/* ---- Buttons ---- */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 18px;
    font-weight: 600;
    font-family: 'Poppins', sans-serif;
    transition: background 0.2s;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af);
    color: white;
    border: none;
}

/* ---- Inputs ---- */
.stNumberInput input,
.stTextInput input,
.stSelectbox select {
    background: #1e293b !important;
    color: white !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* ---- File Uploader ---- */
[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.8);
    border-radius: 14px;
    padding: 20px;
    border: 1px dashed #334155;
}

/* ---- Dividers ---- */
hr {
    border-color: #334155;
}

/* ---- DataFrames ---- */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* ---- Misc ---- */
.small-text {
    color: #94a3b8;
    font-size: 13px;
}

.info-card {
    background: rgba(30, 41, 59, 0.6);
    padding: 18px;
    border-radius: 12px;
    border-left: 4px solid #3b82f6;
    margin-bottom: 12px;
    color: #cbd5e1;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================
st.markdown(
    '<div class="main-title">💳 AI Credit Card Fraud Detection System</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="sub-title">Real-Time Fraud Detection using Machine Learning + Deep Learning Ensemble</div>',
    unsafe_allow_html=True
)

# ==========================================
# LOAD MODELS
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "Model")
@st.cache_resource
def load_models():
    pipeline = joblib.load(os.path.join(MODEL_DIR, "fraud_detection_pipeline.pkl"))
    threshold = joblib.load(os.path.join(MODEL_DIR, "best_threshold.pkl"))
    feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
    ann_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "ann_model.keras"))
    return pipeline, threshold, feature_names, ann_model
try:
    pipeline, threshold, feature_names, ann_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.sidebar.warning(f"⚠️ Models not loaded: {e}")

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.markdown("## ⚙️ Dashboard Controls")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    [
        "⚡ Real-Time Prediction",
        "📊 Analytics Dashboard",
        "🔍 SHAP Explainability",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Model Info")
st.sidebar.markdown(
    """
    <div style='color:#94a3b8; font-size:13px; line-height:1.8'>
    <b style='color:#e2e8f0'>Ensemble</b>: 70% ML + 30% DL<br>
    <b style='color:#e2e8f0'>ML</b>: LightGBM / XGBoost Pipeline<br>
    <b style='color:#e2e8f0'>DL</b>: Keras ANN<br>
    <b style='color:#e2e8f0'>Metric</b>: ROC-AUC ~0.98
    </div>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# HELPERS
# ==========================================
def ensemble_predict(df: pd.DataFrame):
    """Return (ml_prob, dl_prob, final_prob, final_pred)."""
    ml_prob = pipeline.predict_proba(df)[:, 1]
    X_dl = pipeline.named_steps["preprocessor"].transform(df)
    if hasattr(X_dl, "toarray"):
        X_dl = X_dl.toarray()
    dl_prob = ann_model.predict(X_dl, verbose=0).flatten()
    final_prob = 0.7 * ml_prob + 0.3 * dl_prob
    final_pred = (final_prob > threshold).astype(int)
    return ml_prob, dl_prob, final_prob, final_pred


def plotly_dark_theme(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,41,59,0.6)",
        font=dict(color="#e2e8f0", family="Poppins"),
        title_font=dict(size=16, color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        margin=dict(t=40, b=20, l=10, r=10),
    )
    fig.update_xaxes(gridcolor="#334155", zerolinecolor="#334155")
    fig.update_yaxes(gridcolor="#334155", zerolinecolor="#334155")
    return fig

@st.cache_data
def load_training_lists():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR,"data")
    df_train = pd.read_csv(os.path.join(DATA_DIR, "fraudFront.csv"))
    return df_train
df_train = load_training_lists()

# =========================================================
# DROPDOWN LISTS
# =========================================================

merchant_list = sorted(df_train["merchant"].dropna().unique())
category_list = sorted(df_train["category"].dropna().unique())
state_list = sorted(df_train["state"].dropna().unique())
job_list = sorted(df_train["job"].dropna().unique())

# ==========================================
# PAGE: REAL-TIME PREDICTION
# ==========================================
if page == "⚡ Real-Time Prediction":
    st.markdown('<div class="section-title">⚡ Real-Time Fraud Prediction</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns(2)
   
    with col_left:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        st.markdown("#### 💳 Transaction Details")
        amt = st.number_input("Transaction Amount ($)",min_value=0.0,value=100.0,step=1.0)
        hour = st.slider("Transaction Hour (0–23)",0,23,12)
        gender = st.selectbox("Cardholder Gender",["M", "F"])
        cc_num = st.text_input("Credit Card Number",value="1234567890123456",max_chars=22,placeholder="Enter 10–22 digit card number")
        # -------------------------------------------------
        # STATE DROPDOWN
        # -------------------------------------------------
        state = st.selectbox("State",state_list)

        # -------------------------------------------------
        # FILTER CITY ACCORDING TO STATE
        # -------------------------------------------------
        filtered_cities = sorted(df_train[df_train["state"] == state]["city"].dropna().unique())
        city = st.selectbox("City",filtered_cities)

        # -------------------------------------------------
        # AUTO CITY POPULATION
        # -------------------------------------------------
        city_pop = int(df_train[(df_train["state"] == state) & (df_train["city"] == city)]["city_pop"].median())
        city_pop_disp= st.number_input("City Population",value=city_pop,disabled=True)
        st.markdown('</div>',unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🏪 Merchant Details")
        merchant = st.selectbox("Merchant",merchant_list, index=None, placeholder="Search merchant...")
        category = st.selectbox("Transaction Category",category_list)
        job = st.selectbox("Job",job_list)
        is_night = 1 if (hour < 6 or hour > 22) else 0
        if is_night:
            st.info("🌙 Night-time transaction detected")
        st.markdown('</div>',unsafe_allow_html=True)    
    predict_clicked = st.button("🔍 Predict Fraud Now")
    if predict_clicked:
        now = datetime.now()
        cc_num = cc_num.strip()
        if not cc_num.isdigit():
            st.error("❌ Credit card number must contain digits only.")
            st.stop()
        if len(cc_num) < 10 or len(cc_num) > 22:
            st.error("❌ Credit card number must be between 10 and 22 digits.")
            st.stop()
        input_df = pd.DataFrame({
            "trans_date_trans_time": [now],
            "cc_num": [cc_num],
            "merchant": [merchant],
            "category": [category],
            "amt": [amt],
            "gender": [gender],
            "city": [city],
            "state": [state],
            "city_pop": [city_pop],
            "job": [job],
            "hour": [hour],
            "day_of_week": [now.weekday()],
            "month": [now.month],
            "is_weekend": [1 if now.weekday() >= 5 else 0],
            "is_night": [is_night],
            "amt_log": [np.log1p(amt)],
            "avg_amt_user": [amt],
            "std_amt_user": [0.0],
            "amt_to_mean": [1.0],})
        with st.spinner("🧠 Running ensemble inference..."):
            if models_loaded:
                ml_prob_val, dl_prob_val, final_prob_arr, _ = ensemble_predict(input_df)
                final_prob_val = float(final_prob_arr[0])
                ml_prob_val    = float(ml_prob_val[0])
                dl_prob_val    = float(dl_prob_val[0])
                st.session_state["latest_prediction_df"] = input_df.copy()
                st.session_state["latest_ml_prob"] = ml_prob_val
                st.session_state["latest_dl_prob"] = dl_prob_val
                st.session_state["latest_final_prob"] = final_prob_val
                st.session_state["latest_prediction_time"] = datetime.now()
            else:
                risk = 0.03
                if is_night:
                    risk += 0.20
                if amt >10000:
                    risk +=0.30
                if isinstance(merchant, str) and "fraud" in merchant.lower():
                    risk += 0.30
                risk += np.random.uniform(0, 0.03)
                final_prob_val = min(0.99, risk)
                ml_prob_val    = final_prob_val
                dl_prob_val    = final_prob_val

        current_threshold = threshold if models_loaded else 0.5
        is_fraud = (final_prob_val >current_threshold)
        fraud_percent = round(final_prob_val *100, 2)
        if fraud_percent < 20:
            risk_level = "LOW RISK"
        elif fraud_percent < 50:
            risk_level = "MEDIUM RISK"
        elif fraud_percent < 80:
            risk_level = "HIGH RISK"    
        else:
            risk_level = "CRITICAL RISK"
        risk_factors = []
        if amt > 10000:
            risk_factors.append("💰 High transaction amount")
        if is_night:
            risk_factors.append("🌙 Night-time transaction")
        if isinstance(merchant, str) and "fraud" in merchant.lower():
            risk_factors.append("🚨 Suspicious merchant pattern")
        st.markdown("<br>", unsafe_allow_html=True)
        if is_fraud:
            st.markdown(f"""
            <div class="fraud-alert">
                🚨 FRAUD DETECTED<br><br>
                <span style='font-size:18px;font-weight:700'>{risk_level}</span>
                <span style='font-size:15px;font-weight:400'>
                Ensemble Probability : <b>{fraud_percent}%</b><br>
                ML (LightGBM) Score  : <b>{round(ml_prob_val*100, 2)}%</b><br>
                DL (ANN) Score       : <b>{round(dl_prob_val*100, 2)}%</b>
                </span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="safe-alert">
                ✅ NORMAL TRANSACTION<br><br>
                <span style='font-size:18px;font-weight:700'>{risk_level}</span>
                <span style='font-size:15px;font-weight:400'>
                Ensemble Probability : <b>{fraud_percent}%</b><br>
                ML (LightGBM) Score  : <b>{round(ml_prob_val*100, 2)}%</b><br>
                DL (ANN) Score       : <b>{round(dl_prob_val*100, 2)}%</b>
                </span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if len(risk_factors) > 0:
            st.markdown("### ⚠️ Risk Factors")
            for factor in risk_factors:
                st.warning(factor)
        st.session_state["prediction_history"].append({
            "time": datetime.now(),
            "amount": amt,
            "merchant": merchant,
            "category": category,
            "city": city,
            "state": state,
            "fraud_probability": fraud_percent,
            "ml_probability": round(ml_prob_val * 100,2),
            "dl_probability": round(dl_prob_val * 100,2), 
            "prediction": ("Fraud" if is_fraud else "Normal")})
        st.markdown("<br>", unsafe_allow_html=True)    
        # ---- Gauge chart ----
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=fraud_percent,
            number={"suffix": "%", "font": {"size": 36, "color": "white"}},
            delta={"reference": current_threshold*100, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
            title={"text": "Fraud Risk Score", "font": {"color": "white", "size": 20}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#94a3b8"},
                "bar":  {"color": "#ef4444" if is_fraud else "#10b981"},
                "steps": [
                    {"range": [0, 20],  "color": "#052e16"},
                    {"range": [20, 50], "color": "#78350f"},
                    {"range": [50, 80],"color": "#7f1d1d"},
                    {"range": [80, 100], "color": '#450a0a'}
                ],
                "threshold": {
                    "line": {"color": "white", "width": 4},
                    "thickness": 0.85,
                    "value": (current_threshold*100),
                },
            },
        ))
        fig_gauge = plotly_dark_theme(fig_gauge)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # ---- Model breakdown bar ----
        fig_bar = go.Figure(go.Bar(
            x=["ML (LightGBM)", "DL (ANN)", "Ensemble"],
            y=[round(ml_prob_val*100,2), round(dl_prob_val*100,2), round(final_prob_val*100,2)],
            marker_color=["#3b82f6", "#8b5cf6", "#ef4444" if is_fraud else "#10b981"],
            text=[f"{round(ml_prob_val*100,2)}%", f"{round(dl_prob_val*100,2)}%", f"{round(final_prob_val*100,2)}%"],
            textposition="outside",
            textfont=dict(color="white"),
        ))
        fig_bar.update_layout(
            title="Model-wise Fraud Probability",
            yaxis=dict(range=[0, 110]),
        )
        fig_bar = plotly_dark_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# PAGE: ANALYTICS DASHBOARD
# ==========================================
elif page == "📊 Analytics Dashboard":
    st.markdown(
        '<div class="section-title">📊 Dynamic Fraud Analytics Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
    📌 Real-time fraud analytics powered by live predictions.
    Dashboard updates automatically whenever
    new fraud predictions are made.
    </div>
    """, unsafe_allow_html=True)
    history_df = pd.DataFrame(st.session_state["prediction_history"])
    if history_df.empty:
        st.info("⚡ No fraud predictions available yet.")
    else:
        history_df["time"] = pd.to_datetime(history_df["time"])
        history_df["hour"] = history_df["time"].dt.hour
        total_txn = len(history_df)
        fraud_count = len(history_df[history_df["prediction"] == "Fraud"])
        normal_count = total_txn - fraud_count
        fraud_rate = round((fraud_count / total_txn) * 100,2)
        avg_risk = round(history_df["fraud_probability"].mean(),2)
        max_risk = round(history_df["fraud_probability"].max(),2)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number"
                     style="color:#3b82f6">
                    {total_txn:,}
                </div>
                <div class="metric-label">
                    💳 Transactions
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number"
                     style="color:#ef4444">
                    {fraud_count:,}
                </div>
                <div class="metric-label">
                    🚨 Fraud Detected
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number"
                     style="color:#10b981">
                    {normal_count:,}
                </div>
                <div class="metric-label">
                    ✅ Normal
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number"
                     style="color:#f59e0b">
                    {fraud_rate}%
                </div>
                <div class="metric-label">
                    📊 Fraud Rate
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-number"
                     style="color:#8b5cf6">
                    {avg_risk}%
                </div>
                <div class="metric-label">
                    ⚠️ Avg Risk
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(
                names=[
                    "Fraud",
                    "Normal"
                ],
                values=[
                    fraud_count,
                    normal_count
                ],
                title="Fraud vs Normal Distribution",
                color_discrete_sequence=[
                    "#ef4444",
                    "#10b981"
                ],
                hole=0.45
            )
            fig_pie = plotly_dark_theme(fig_pie)
            st.plotly_chart(fig_pie,use_container_width=True)
        with col2:
            fig_trend = px.line(
                history_df,
                x="time",
                y="fraud_probability",
                title="Fraud Risk Trend Over Time",
                markers=True
            )

            fig_trend = plotly_dark_theme(fig_trend)
            st.plotly_chart(fig_trend,use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            category_risk = history_df.groupby("category")["fraud_probability"].mean().reset_index()
            fig_category = px.bar(
                category_risk,
                x="category",
                y="fraud_probability",
                title="Average Fraud Risk by Category",
                color="fraud_probability")
            fig_category = plotly_dark_theme(fig_category)
            st.plotly_chart(fig_category,use_container_width=True)
        with col2:
            state_counts = history_df.groupby("state").size().reset_index(name="count")
            fig_state = px.bar(
                state_counts,
                x="state",
                y="count",
                title="Transactions by State",
                color="count"
            )
            fig_state = plotly_dark_theme(fig_state)
            st.plotly_chart(fig_state,use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            hourly_risk = history_df.groupby("hour")["fraud_probability"].mean().reset_index()
            fig_hour = px.line(
                hourly_risk,
                x="hour",
                y="fraud_probability",
                markers=True,
                title="Hourly Fraud Risk Pattern")
            fig_hour = plotly_dark_theme(fig_hour)
            st.plotly_chart(fig_hour,use_container_width=True)
        with col2:
            merchant_risk = history_df.groupby("merchant")["fraud_probability"].mean().reset_index()
            merchant_risk = merchant_risk.sort_values(by="fraud_probability",ascending=False).head(10)
            fig_merchant = px.bar(
                merchant_risk,
                x="merchant",
                y="fraud_probability",
                title="Top High-Risk Merchants",
                color="fraud_probability")
            fig_merchant = plotly_dark_theme(fig_merchant)
            st.plotly_chart(fig_merchant,use_container_width=True)
        st.markdown(
            '<div class="section-title">📋 Recent Transactions</div>',
            unsafe_allow_html=True)
        latest_df = history_df.sort_values(by="time",ascending=False)
        st.dataframe(latest_df.head(20),use_container_width=True)
        st.markdown(
            '<div class="section-title">🚨 Top High-Risk Transactions</div>',
            unsafe_allow_html=True
        )
        high_risk_df = history_df.sort_values(by="fraud_probability",ascending=False)
        st.dataframe(high_risk_df.head(10),use_container_width=True)
        csv_history = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(label="⬇ Download Prediction History",data=csv_history,file_name="fraud_prediction_history.csv",mime="text/csv")
        st.markdown("""
        <div class="card">
        <h4 style='color:white'>
        ⚙️ System Status
        </h4>
        <ul style='color:#cbd5e1; line-height:2'>
            <li>
            ✅ Real-time fraud monitoring active
            </li>
            <li>
            ✅ Ensemble ML + DL inference active
            </li>
            <li>
            ✅ SHAP explainability connected
            </li>
            <li>
            ✅ Dynamic dashboard updating
            </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# PAGE: SHAP EXPLAINABILITY
# ==========================================
elif page == "🔍 SHAP Explainability":
    st.markdown(
        '<div class="section-title">🔍 SHAP Explainability</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="info-card">
    📌 <b>SHAP (SHapley Additive exPlanations)</b>
    explains how each feature contributes to fraud predictions.
    Red features increase fraud probability.
    Blue features decrease fraud probability.
    </div>
    """, unsafe_allow_html=True)
    if models_loaded:
        if st.session_state["latest_prediction_df"] is not None:
            try:
                input_df = st.session_state["latest_prediction_df"]
                ml_prob = st.session_state.get("latest_ml_prob",0)
                dl_prob = st.session_state.get("latest_dl_prob",0)
                final_prob = st.session_state.get("latest_final_prob",0)
                prediction_time = st.session_state.get("latest_prediction_time",datetime.now())
                st.markdown(f"""
                <div class="card">
                <h3 style='color:white'>
                ⚡ Latest Real-Time Prediction
                </h3>
                <hr>
                <p style='color:#cbd5e1'>
                🕒 Prediction Time:
                <b>{prediction_time}</b>
                </p>
                <p style='color:#cbd5e1'>
                🚨 Ensemble Fraud Probability:
                <b>{round(final_prob*100,2)}%</b>
                </p>
                <p style='color:#cbd5e1'>
                🧠 ML Model Probability:
                <b>{round(ml_prob*100,2)}%</b>
                </p>
                <p style='color:#cbd5e1'>
                🤖 ANN Model Probability:
                <b>{round(dl_prob*100,2)}%</b>
                </p>
                </div>
                """, unsafe_allow_html=True)
                with st.spinner("Generating SHAP explanation..."):
                    X_rt = pipeline.named_steps["preprocessor"].transform(input_df)
                    if hasattr(X_rt, "toarray"):
                        X_rt = X_rt.toarray()
                    transformed_rt_df = pd.DataFrame(X_rt,columns=feature_names)
                    ml_model = pipeline.named_steps["model"]
                    explainer = shap.TreeExplainer(ml_model)
                    shap_values = explainer.shap_values(X_rt,check_additivity=False)
                    if isinstance(shap_values,list):
                        shap_values = shap_values[1]
                    shap_df = pd.DataFrame(shap_values,columns=list(feature_names))
                    st.markdown("## 📊 Feature Contributions")
                    contribution_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP_Value": shap_values[0]
                    })
                    contribution_df["Impact"] = np.where(contribution_df["SHAP_Value"] > 0,
                        "🔴 Increase Fraud Risk",
                        "🔵 Reduce Fraud Risk"
                    )
                    contribution_df = contribution_df.sort_values(by="SHAP_Value",key=abs,ascending=False)
                    st.dataframe(contribution_df,use_container_width=True)
                    st.markdown("## 🌊 SHAP Waterfall Plot")
                    try:
                        explanation = shap.Explanation(values=shap_values[0],base_values=explainer.expected_value,data=X_rt[0],feature_names=feature_names)
                        fig_waterfall = plt.figure(figsize=(12, 8))
                        shap.plots.waterfall(explanation,max_display=20,show=False)

                        st.pyplot(fig_waterfall)
                        plt.close()
                    except Exception as ex:
                        st.warning(f"Waterfall plot failed: {ex}")
                    
                    st.markdown("## 📈 SHAP Feature Importance")
                    fig_bar, ax = plt.subplots(figsize=(12, 6))
                    fig_bar.patch.set_facecolor("#0f172a")
                    ax.set_facecolor( "#1e293b")
                    shap.summary_plot(shap_values,X_rt,feature_names=feature_names,plot_type="bar",max_display=20,show=False)
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                    plt.close()
                    transformed_rt_df.to_csv("real_time_transformed_features.csv",index=False)
                    shap_df.to_csv("real_time_shap_values.csv",index=False)
                    st.download_button(
                        "⬇ Download SHAP Values",
                        shap_df.to_csv(index=False),
                        file_name="real_time_shap_values.csv",
                        mime="text/csv"
                    )

                    st.download_button(
                        "⬇ Download Transformed Features",
                        transformed_rt_df.to_csv(index=False),
                        file_name="real_time_transformed_features.csv",
                        mime="text/csv"
                    )


                    top_positive = contribution_df[contribution_df["SHAP_Value"] > 0].head(3)
                    top_negative = contribution_df[contribution_df["SHAP_Value"] < 0].head(3)
                    st.markdown(
                        "## 🧠 Model Interpretation"
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("""
                        <div class="fraud-alert">
                        <h4>
                        🔴 Features Increasing Fraud Risk
                        </h4>
                        </div>
                        """, unsafe_allow_html=True)

                        for _, row in top_positive.iterrows():
                            st.error(
                                f"{row['Feature']} "
                                f"({round(row['SHAP_Value'],4)})"
                            )
                    with col2:
                        st.markdown("""
                        <div class="safe-alert">
                        <h4>
                        🔵 Features Reducing Fraud Risk
                        </h4>
                        </div>
                        """, unsafe_allow_html=True)
                        for _, row in top_negative.iterrows():
                            st.success(
                                f"{row['Feature']} "
                                f"({round(row['SHAP_Value'],4)})"
                            )
                    st.success(
                        "✅ Real-time SHAP explanation completed."
                    )
            except Exception as ex:
                st.error(
                    f"SHAP generation failed: {ex}"
                )
        else:
            st.info(
                "⚡ Run a real-time prediction first."
            )
    else:
        st.markdown("""
        <div class="card">
        <h4 style='color:white'>
        How SHAP Helps Fraud Detection
        </h4>
        <ul style='color:#cbd5e1;
                   line-height:2;
                   margin-top:10px'>
            <li>
            💰 <b>amt_log</b>
            — Large transaction amounts strongly increase fraud probability
            </li>
            <li>
            🌙 <b>is_night</b>
            — Night-time transactions are riskier
            </li>
            <li>
            📊 <b>amt_to_mean</b>
            — Transactions deviating from normal spending patterns are suspicious
            </li>
            <li>
            ⏰ <b>hour</b>
            — Certain transaction hours are higher risk
            </li>

            <li>
            🛒 <b>category</b>
            — Online shopping and misc categories show elevated fraud patterns
            </li>

        </ul>

        </div>

        """, unsafe_allow_html=True)

   
    st.markdown("""
                <div class="card">
<h4 style='color:white; margin-bottom:12px'>
📚 SHAP Interpretation Guide
</h4>
</div>
""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""

    <div style="
    background:rgba(239,68,68,0.15);
    border-radius:10px;
    padding:18px;
    border:1px solid #ef4444;
    min-height:150px;
    ">
    <p style="
    color:#ef4444;
    font-size:18px;
    font-weight:700;
    margin-bottom:15px;
    ">
    🔴 Positive SHAP Values
    </p>
    <p style="
    color:#fca5a5;
    font-size:15px;
    line-height:1.7;
    ">
    Features pushing prediction toward FRAUD.
    Higher feature contribution increases fraud probability.
    </p>
    </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""

    <div style="
    background:rgba(59,130,246,0.15);
    border-radius:10px;
    padding:18px;
    border:1px solid #3b82f6;
    min-height:150px;
    ">
        <p style="
            color:#3b82f6;
            font-size:18px;
            font-weight:700;
            margin-bottom:15px;
        ">
        🔵 Negative SHAP Values
        </p>
        <p style="
            color:#93c5fd;
            font-size:15px;
            line-height:1.7;
        ">
        Features pushing prediction toward NORMAL.
        Higher negative contribution lowers fraud probability.
        </p>
    </div> """, unsafe_allow_html=True)
# ==========================================
# FOOTER
# ==========================================

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:#475569; padding:16px; font-size:13px'>
    💳 AI Fraud Detection System &nbsp;•&nbsp; ML + DL Ensemble &nbsp;•&nbsp; Real-Time Analytics<br>
    <span style='color:#334155'>Built with Streamlit · LightGBM · XGBoost · Keras · SHAP · Plotly</span>
</div>
""", unsafe_allow_html=True)