import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb  # For XGBoost fallback
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(page_title="DPR AI AI Assessor - MDoNER", layout="wide")
st.title("🚀 AI-Powered DPR Quality Assessment & Risk Prediction")
st.markdown("Upload a DPR PDF for instant analysis. Powered by ML models for NE projects.")

# Model options
MODEL_OPTIONS = ['Random Forest', 'XGBoost', 'Gradient Boosting']
DEFAULT_MODEL = 'XGBoost'  # Recommended for risk prediction

# Load preprocessor & models (dynamic loading)
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        # Load all models
        quality_models = {
            'Random Forest': joblib.load('randomforest_quality_reg.pkl'),
            'XGBoost': joblib.load('xgboost_quality_reg.pkl'),
            'Gradient Boosting': joblib.load('gradientboosting_quality_reg.pkl')
        }
        risk_models = {
            'Random Forest': joblib.load('randomforest_risk_reg.pkl'),
            'XGBoost': joblib.load('xgboost_risk_reg.pkl'),
            'Gradient Boosting': joblib.load('gradientboosting_risk_reg.pkl')
        }
        # Load medians for defaults
        train_df = pd.read_csv('cleaned_dpr_train.csv')
        numeric_medians = train_df[['funding_amount_cr', 'prior_experience_years', 'estimated_duration_months', 'actual_duration_months', 'delay']].median().to_dict()
        return preprocessor, quality_models, risk_models, numeric_medians
    except FileNotFoundError as e:
        st.warning(f"Some models missing ({e})—using rule-based fallback. Run train_models.py!")
        return None, {}, {}, {}

preprocessor, quality_models, risk_models, numeric_medians = load_models()

# Parsing function (unchanged from last)
def parse_dpr(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text().lower()
    
    project_match = re.search(r'(road|power|tourism|health|education|watershed)', full_text)
    project_type = project_match.group(1).capitalize() if project_match else 'Unknown'
    
    location_match = re.search(r'(assam|arunachal pradesh|manipur|meghalaya|mizoram|nagaland|sikkim|tripura|dima hasao)', full_text)
    location = location_match.group(1).title() if location_match else 'Unknown'
    
    budget_match = re.search(r'rs\.?\s*?(\d+\.?\d*)\s*cr|(\d+\.?\d*)\s*cr', full_text)
    funding_amount_cr = float(budget_match.group(1) or budget_match.group(2)) if budget_match else 10.0
    
    duration_match = re.search(r'(\d+\.?\d*)\s*months?', full_text)
    estimated_duration_months = float(duration_match.group(1)) if duration_match else 12.0
    
    prior_experience_years = 5.0
    delay = 0
    approval_outcome = 'Unknown'
    issues_found = 'None'
    if re.search(r'(budget miscalculation|environmental non-compliance|timeline unrealistic|technical flaw)', full_text):
        issues_found = re.search(r'(budget miscalculation|environmental non-compliance|timeline unrealistic|technical flaw)', full_text).group(1)
    
    actual_duration_months = estimated_duration_months
    
    sections = {
        'project_type': project_type,
        'location': location,
        'funding_amount_cr': funding_amount_cr,
        'estimated_duration_months': estimated_duration_months,
        'prior_experience_years': prior_experience_years,
        'actual_duration_months': actual_duration_months,
        'delay': delay,
        'approval_outcome': approval_outcome,
        'issues_found': issues_found
    }
    return sections, full_text

# Feature extraction (unchanged)
def extract_features(sections):
    expected_cols = [
        'project_type', 'location', 'funding_amount_cr', 'prior_experience_years',
        'estimated_duration_months', 'actual_duration_months', 'delay',
        'approval_outcome', 'issues_found'
    ]
    
    feature_data = {col: sections.get(col, numeric_medians.get(col, 0) if col in ['funding_amount_cr', 'prior_experience_years', 'estimated_duration_months', 'actual_duration_months', 'delay'] else 'Unknown') for col in expected_cols}
    df_feats = pd.DataFrame([feature_data])
    
    numeric_cols = ['funding_amount_cr', 'prior_experience_years', 'estimated_duration_months', 'actual_duration_months', 'delay']
    categorical_cols = ['project_type', 'location', 'approval_outcome', 'issues_found']
    
    if preprocessor:
        try:
            feats_transformed = preprocessor.transform(df_feats)
            return feats_transformed
        except ValueError as e:
            if "columns are missing" in str(e):
                st.error(f"Feature mismatch: {e}. Using fallback.")
                # Get expected shape from preprocessor
                return np.zeros((1, preprocessor.n_features_in_))
            raise e
    else:
        return np.array([[0.0] * len(numeric_cols) + [0.0] * len(categorical_cols)])

# ML Predictions (now model-specific)
def predict_quality(features, model_name):
    if model_name in quality_models:
        score = quality_models[model_name].predict(features)[0]
        return max(0, min(10, score))
    else:
        # Rule-based fallback
        budget = features[0][0] if len(features) > 0 and len(features[0]) > 0 else 10
        score = 7.0 if budget > 5 else 4.0
        return score

def predict_risk(features, model_name):
    if model_name in risk_models:
        overrun_pct = risk_models[model_name].predict(features)[0]
        level = "High" if overrun_pct > 20 else "Medium" if overrun_pct > 10 else "Low"
        return {"level": level, "overrun_pct": max(0, round(overrun_pct, 1))}
    else:
        budget = features[0][0] if len(features) > 0 and len(features[0]) > 0 else 10
        overrun_pct = 15.0 if budget > 20 else 5.0
        level = "Medium" if overrun_pct > 10 else "Low"
        return {"level": level, "overrun_pct": overrun_pct}

# Main App
uploaded_file = st.file_uploader("📁 Upload DPR PDF", type="pdf")

if uploaded_file is not None:
    # Model selection below upload
    selected_model = st.selectbox("🤖 Select ML Model", MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
    st.info(f"Using: **{selected_model}** – XGBoost recommended for accurate risk prediction.")
    
    with st.spinner("Parsing PDF..."):
        sections, full_text = parse_dpr(uploaded_file.read())
    features = extract_features(sections)
    
    # Predictions with selected model
    quality_score = predict_quality(features, selected_model)
    risks = predict_risk(features, selected_model)
    
    # Display
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Quality Score", f"{quality_score:.1f}/10", delta=f"{quality_score - 5:+.1f}")
        st.metric("Risk Level", risks['level'], delta=f"{risks['overrun_pct']}%")
    with col2:
        st.metric("Predicted Overrun", f"{risks['overrun_pct']}%", delta=None)
        st.success("✅ Analysis Complete!") if quality_score >= 7 else st.warning("⚠️ Needs Revision")
    
    # Rest unchanged: Sections, Chart, Text, Recommendations
    st.subheader("📋 Extracted Sections")
    sections_df = pd.DataFrame(list(sections.items()), columns=['Key', 'Value'])
    st.table(sections_df)
    
    st.subheader("🎯 Risk Breakdown")
    fig = px.pie(values=[risks['overrun_pct'], 100 - risks['overrun_pct']], names=['Predicted Overrun', 'Baseline'], 
                 title="Cost Overrun Risk vs. Expected")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📄 DPR Text Preview")
    st.text_area("", full_text[:1000], height=200, disabled=True)
    
    st.subheader("💡 Recommendations")
    if quality_score < 5:
        st.error("- Revise budget and environmental sections.")
    elif risks['level'] == 'High':
        st.warning("- Add contingency for overruns (e.g., monsoon risks).")
    else:
        st.info("- Proceed to approval with minor tweaks.")

else:
    st.info("👆 Upload a PDF to get started. Try the Dima Hasao sample!")

# Footer
st.markdown("---")
st.markdown("*Built for SIH 2025 - MDoNER | xAI Grok Assisted*")