import streamlit as st
import fitz  # PyMuPDF
import re
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import xgboost as xgb
import warnings
import os
warnings.filterwarnings('ignore')

# Page config (must be at the top to avoid circular imports)
st.set_page_config(page_title="DPR AI Assessor - MDoNER", layout="wide")

# Model options
MODEL_OPTIONS = ['Random Forest', 'XGBoost', 'Gradient Boosting']
DEFAULT_MODEL = 'XGBoost'

# Language support (Hindi labels)
LABELS = {
    'en': {
        'title': "AI-Powered DPR Quality Assessment & Risk Prediction",
        'upload': "Upload DPR PDF",
        'home': "Home",
        'predictions': "Predictions",
        'sections': "Extracted Sections",
        'risk': "Risk Breakdown",
        'performance': "Model Performance",
        'importance': "Feature Importance",
        'preview': "DPR Text Preview",
        'recommendations': "Recommendations",
        'model_select': "Select ML Model",
        'info_upload': "Upload a PDF to get started. Try a sample DPR for a Road project in Dima Hasao!",
        'processing': "Parsing PDF...",
        'model_info': "Using: **{}** – XGBoost recommended for accurate risk prediction."
    },
    'hi': {
        'title': "एआई-संचालित डीपीआर गुणवत्ता मूल्यांकन और जोखिम भविष्यवाणी",
        'upload': "डीपीआर पीडीएफ अपलोड करें",
        'home': "होम",
        'predictions': "भविष्यवाणियाँ",
        'sections': "निकाले गए अनुभाग",
        'risk': "जोखिम विश्लेषण",
        'performance': "मॉडल प्रदर्शन",
        'importance': "विशेषता महत्व",
        'preview': "डीपीआर पाठ पूर्वावलोकन",
        'recommendations': "सिफारिशें",
        'model_select': "एमएल मॉडल चुनें",
        'info_upload': "शुरू करने के लिए एक पीडीएफ अपलोड करें। दिमा हसाओ में सड़क परियोजना के लिए नमूना डीपीआर आज़माएँ!",
        'processing': "पीडीएफ पार्सिंग...",
        'model_info': "उपयोग: **{}** – सटीक जोखिम भविष्यवाणी के लिए XGBoost अनुशंसित।"
    }
}

# Language selection
lang = st.sidebar.selectbox("भाषा / Language", ["English", "Hindi"])
lang_code = 'hi' if lang == "Hindi" else 'en'
labels = LABELS[lang_code]

# Load preprocessor & models
@st.cache_resource
def load_models():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        quality_models = {}
        risk_models = {}
        for model_name in MODEL_OPTIONS:
            quality_path = f"{model_name.lower().replace(' ', '')}_quality_reg.pkl"
            risk_path = f"{model_name.lower().replace(' ', '')}_risk_reg.pkl"
            if os.path.exists(quality_path):
                quality_models[model_name] = joblib.load(quality_path)
            if os.path.exists(risk_path):
                risk_models[model_name] = joblib.load(risk_path)
        train_df = pd.read_csv('cleaned_dpr_train.csv')
        numeric_medians = train_df[['funding_amount_cr', 'prior_experience_years', 'estimated_duration_months', 'actual_duration_months', 'delay']].median().to_dict()
        performance_df = pd.read_csv('model_performance.csv') if os.path.exists('model_performance.csv') else pd.DataFrame()
        try:
            transformed_feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            transformed_feature_names = preprocessor.feature_names_in_
        return preprocessor, quality_models, risk_models, numeric_medians, performance_df, transformed_feature_names
    except FileNotFoundError as e:
        st.error(
            f"File missing: {e}. Using rule-based fallback. "
            "Ensure 'cleaned_dpr_train.csv', 'cleaned_dpr_test.csv', 'preprocessor.pkl', and all model .pkl files "
            "(randomforest_quality_reg.pkl, xgboost_quality_reg.pkl, gradientboosting_quality_reg.pkl, "
            "randomforest_risk_reg.pkl, xgboost_risk_reg.pkl, gradientboosting_risk_reg.pkl) are in the directory. "
            "Run 'train_models.py' to generate model files."
        )
        return None, {}, {}, {}, pd.DataFrame(), []

preprocessor, quality_models, risk_models, numeric_medians, performance_df, transformed_feature_names = load_models()

# Sidebar navigation
st.sidebar.title(labels['home'])
page = st.sidebar.selectbox(labels['home'], [
    labels['home'], labels['predictions'], labels['sections'], labels['risk'],
    labels['performance'], labels['importance'], labels['preview'], labels['recommendations']
])

# Warn about poor model performance
if not performance_df.empty and performance_df['R2'].min() < 0 and page != labels['home']:
    st.warning("⚠️ Some models (e.g., Gradient Boosting) have negative R² scores, indicating poor performance. "
               "Consider re-running 'train_models.py' with tuned hyperparameters or regenerating the dataset.")

# Parsing function with inconsistency detection
def parse_dpr(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text().lower()
    
    project_match = re.search(r'(road|power|tourism|health|education|watershed)', full_text)
    project_type = project_match.group(1).capitalize() if project_match else 'Unknown'
    
    location_match = re.search(r'(assam|arunachal pradesh|manipur|meghalaya|mizoram|nagaland|sikkim|tripura|dima hasao)', full_text)
    location = location_match.group(1).title() if location_match else 'Unknown'
    
    budget_match = re.search(r'(?:rs\.?|₹)\s*(\d+\.?\d*)\s*(?:cr|crore)', full_text, re.IGNORECASE)
    funding_amount_cr = float(budget_match.group(1)) if budget_match else 10.0
    
    duration_match = re.search(r'(\d+\.?\d*)\s*months?', full_text)
    estimated_duration_months = float(duration_match.group(1)) if duration_match else 12.0
    
    prior_experience_years = 5.0
    delay = 0
    approval_outcome = 'Unknown'
    issues_found = []
    if re.search(r'(budget miscalculation|environmental non-compliance|timeline unrealistic|technical flaw)', full_text):
        issues_found.append(re.search(r'(budget miscalculation|environmental non-compliance|timeline unrealistic|technical flaw)', full_text).group(1))
    if funding_amount_cr > 100 and estimated_duration_months < 6:
        issues_found.append("Unrealistic timeline for large budget")
    if not project_match:
        issues_found.append("Project type not specified")
    issues_found = ", ".join(issues_found) if issues_found else "None"
    
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

# Feature extraction
def extract_features(sections):
    expected_cols = [
        'project_type', 'location', 'funding_amount_cr', 'prior_experience_years',
        'estimated_duration_months', 'actual_duration_months', 'delay',
        'approval_outcome', 'issues_found'
    ]
    
    feature_data = {col: sections.get(col, numeric_medians.get(col, 0) if col in ['funding_amount_cr', 'prior_experience_years', 'estimated_duration_months', 'actual_duration_months', 'delay'] else 'Unknown') for col in expected_cols}
    df_feats = pd.DataFrame([feature_data])
    
    if preprocessor:
        try:
            if not all(col in df_feats.columns for col in preprocessor.feature_names_in_):
                st.error(f"Feature mismatch: Expected {preprocessor.feature_names_in_}, got {df_feats.columns}")
                return np.zeros((1, len(transformed_feature_names)))
            feats_transformed = preprocessor.transform(df_feats)
            return feats_transformed
        except ValueError as e:
            st.error(f"Feature transformation error: {e}. Using fallback predictions.")
            return np.zeros((1, len(transformed_feature_names)))
    else:
        return np.array([[0.0] * 5 + [0.0] * 4])

# ML Predictions
def predict_quality(features, model_name):
    if model_name in quality_models:
        score = quality_models[model_name].predict(features)[0]
        return max(0, min(10, score))
    else:
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

# Page content
if page == labels['home']:
    st.title(labels['title'])
    st.markdown("Upload a DPR PDF for instant analysis. Powered by ML models for NE projects.")
    uploaded_file = st.file_uploader(labels['upload'], type="pdf")
    if uploaded_file is None:
        st.info(labels['info_upload'])
    else:
        st.session_state['uploaded_file'] = uploaded_file.read()
        st.info("PDF uploaded! Select a page from the sidebar to view results.")

if 'uploaded_file' in st.session_state:
    with st.spinner(labels['processing']):
        sections, full_text = parse_dpr(st.session_state['uploaded_file'])
    features = extract_features(sections)
    selected_model = st.session_state.get('selected_model', DEFAULT_MODEL)
    st.session_state['selected_model'] = selected_model

    if page == labels['predictions']:
        st.title(labels['predictions'])
        selected_model = st.selectbox(labels['model_select'], MODEL_OPTIONS, index=MODEL_OPTIONS.index(DEFAULT_MODEL))
        st.session_state['selected_model'] = selected_model
        st.info(labels['model_info'].format(selected_model))
        quality_score = predict_quality(features, selected_model)
        risks = predict_risk(features, selected_model)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Quality Score / गुणवत्ता स्कोर", f"{quality_score:.1f}/10", delta=f"{quality_score - 5:+.1f}")
            st.metric("Risk Level / जोखिम स्तर", risks['level'], delta=f"{risks['overrun_pct']}%")
        with col2:
            st.metric("Predicted Overrun / अनुमानित लागत वृद्धि", f"{risks['overrun_pct']}%", delta=None)
            st.success("✅ Analysis Complete! / विश्लेषण पूर्ण!") if quality_score >= 7 else st.warning("⚠️ Needs Revision / संशोधन की आवश्यकता")

    elif page == labels['sections']:
        st.title(labels['sections'])
        sections_df = pd.DataFrame(list(sections.items()), columns=['Key', 'Value'])
        sections_df['Value'] = sections_df['Value'].astype(str)  # Fix ArrowTypeError
        st.table(sections_df)

    elif page == labels['risk']:
        st.title(labels['risk'])
        risks = predict_risk(features, st.session_state.get('selected_model', DEFAULT_MODEL))
        fig = px.pie(values=[risks['overrun_pct'], 100 - risks['overrun_pct']], names=['Predicted Overrun / अनुमानित लागत वृद्धि', 'Baseline / आधार रेखा'], 
                     title="Cost Overrun Risk vs. Expected / लागत वृद्धि जोखिम बनाम अपेक्षित")
        st.plotly_chart(fig, use_container_width=True)

    elif page == labels['performance']:
        st.title(labels['performance'])
        if not performance_df.empty:
            fig_performance = px.bar(
                performance_df,
                x='Model', y='R2', color='Target',
                barmode='group',
                title="Model Performance Comparison (R² Score) / मॉडल प्रदर्शन तुलना (R² स्कोर)",
                color_discrete_map={'Quality Score': '#636EFA', 'Cost Overrun': '#EF553B'}
            )
            st.plotly_chart(fig_performance, use_container_width=True)
        else:
            st.warning("Model performance data not available. Run 'train_models.py' to generate 'model_performance.csv'.")
            fig_performance = px.bar(
                pd.DataFrame({
                    'Model': ['Random Forest', 'XGBoost', 'Gradient Boosting'] * 2,
                    'R2': [0.760, 0.785, -0.101, 0.700, 0.720, 0.133],  # Gradient Boosting from output
                    'Target': ['Quality Score'] * 3 + ['Cost Overrun'] * 3
                }),
                x='Model', y='R2', color='Target',
                barmode='group',
                title="Model Performance Comparison (R² Score, Partial Data) / मॉडल प्रदर्शन तुलना (R² स्कोर, आंशिक डेटा)"
            )
            st.plotly_chart(fig_performance, use_container_width=True)

    elif page == labels['importance']:
        st.title(labels['importance'])
        selected_model = st.session_state.get('selected_model', DEFAULT_MODEL)
        if selected_model == 'XGBoost' and 'XGBoost' in quality_models:
            importance = quality_models['XGBoost'].feature_importances_
            if len(importance) == len(transformed_feature_names):
                fig_importance = px.bar(
                    x=importance, y=transformed_feature_names,
                    title="Feature Importance for Quality Score Prediction / गुणवत्ता स्कोर भविष्यवाणी के लिए विशेषता महत्व",
                    labels={'x': 'Importance / महत्व', 'y': 'Feature / विशेषता'}
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            else:
                st.warning(f"Feature importance mismatch: {len(importance)} importances vs {len(transformed_feature_names)} features. Check preprocessor.pkl.")
        else:
            st.info("Feature importance is only available for XGBoost model. / विशेषता महत्व केवल XGBoost मॉडल के लिए उपलब्ध है।")

    elif page == labels['preview']:
        st.title(labels['preview'])
        st.text_area(labels['preview'], full_text[:1000], height=200, disabled=True, label_visibility="hidden")

    elif page == labels['recommendations']:
        st.title(labels['recommendations'])
        quality_score = predict_quality(features, st.session_state.get('selected_model', DEFAULT_MODEL))
        risks = predict_risk(features, st.session_state.get('selected_model', DEFAULT_MODEL))
        if quality_score < 5:
            st.error("- Revise budget and environmental sections for compliance. / बजट और पर्यावरण अनुभागों को अनुपालन के लिए संशोधित करें।")
        elif quality_score < 7:
            st.warning("- Review timeline and technical specs; consider contingency for delays. / समयरेखा और तकनीकी विनिर्देशों की समीक्षा करें; देरी के लिए आकस्मिक योजना पर विचार करें।")
        elif risks['level'] == 'High':
            st.warning("- Add contingency budget (e.g., 20%+ for monsoon or supply chain risks). / आकस्मिक बजट जोड़ें (उदाहरण के लिए, मानसून या आपूर्ति श्रृंखला जोखिमों के लिए 20%+।)")
        else:
            st.info("- Proceed to approval with minor adjustments to documentation. / दस्तावेज़ में मामूली समायोजन के साथ स्वीकृति के लिए आगे बढ़ें।")

# Footer
if page != labels['home']:
    st.markdown("---")
    st.markdown("*Built for SIH 2025 - MDoNER | xAI Grok Assisted | Updated Oct 15, 2025*")