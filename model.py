import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load cleaned data (or generate synthetic for demo - comment out for real)
try:
    train_df = pd.read_csv('cleaned_dpr_train.csv')
    test_df = pd.read_csv('cleaned_dpr_test.csv')
    print("Loaded cleaned data.")
except FileNotFoundError:
    # Fallback: Generate synthetic (from earlier scripts)
    np.random.seed(42)
    ne_states = ['Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Sikkim', 'Tripura']
    project_types = ['Road', 'Power', 'Tourism', 'Health', 'Education']
    n_samples = 500
    data = []
    for i in range(n_samples):
        project_type = np.random.choice(project_types)
        location = np.random.choice(ne_states)
        funding_amount = np.random.uniform(1, 100)
        prior_experience = np.random.choice([0, 1, 2, 3, 5, 10])
        estimated_duration = np.random.uniform(6, 48)
        overrun_factor = 1.5 if prior_experience < 2 else 1.0
        cost_overrun_pct = np.random.uniform(0, 50 * overrun_factor) if np.random.rand() > 0.3 else 0
        delay_prob = 0.6 if prior_experience < 2 else 0.4
        delay = np.random.choice([True, False], p=[delay_prob, 1 - delay_prob])
        actual_duration = estimated_duration * (1.2 if delay else 1) + np.random.uniform(-2, 5)
        approval_outcome = np.random.choice(['Approved', 'Rejected', 'Revised'], p=[0.7, 0.1, 0.2])
        possible_issues = ['Budget Miscalculation', 'Environmental Non-Compliance', 'Timeline Unrealistic', 'Technical Flaw']
        num_issues = np.random.randint(0, 4)
        issues = np.random.choice(possible_issues, size=num_issues, replace=False).tolist()
        issues_found = '; '.join(issues) if issues else 'None'
        quality_score = np.random.uniform(4, 10) + (funding_amount / 100) * 0.5
        quality_score = min(10, quality_score)
        if quality_score >= 7:
            quality_category = 'Good'
        elif quality_score >= 5:
            quality_category = 'Needs Revision'
        else:
            quality_category = 'Poor'
        data.append({
            'project_type': project_type, 'location': location, 'funding_amount_cr': round(funding_amount, 2),
            'prior_experience_years': prior_experience, 'estimated_duration_months': round(estimated_duration, 1),
            'actual_duration_months': round(actual_duration, 1), 'cost_overrun_pct': round(cost_overrun_pct, 2),
            'delay': int(delay), 'approval_outcome': approval_outcome, 'issues_found': issues_found,
            'quality_score': round(quality_score, 2), 'quality_category': quality_category
        })
    df = pd.DataFrame(data)
    # Quick clean
    df['issues_found'] = df['issues_found'].fillna('None')
    numeric_cols = ['funding_amount_cr', 'prior_experience_years', 'estimated_duration_months', 'actual_duration_months', 'delay']
    for col in numeric_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5*IQR, Q3 + 1.5*IQR)
    # Preprocess (simplified - assumes scaled; in real, use your preprocessor.pkl)
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    categorical_cols = ['project_type', 'location', 'approval_outcome', 'issues_found']
    numeric_features = numeric_cols
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ])
    feature_cols = numeric_cols + categorical_cols
    X_processed = preprocessor.fit_transform(df[feature_cols])
    cat_encoder = preprocessor.named_transformers_['cat']
    cat_features = cat_encoder.get_feature_names_out(categorical_cols)
    all_features = numeric_features + list(cat_features)
    X_df = pd.DataFrame(X_processed, columns=all_features)
    # Split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(X_df.assign(quality_score=df['quality_score'], cost_overrun_pct=df['cost_overrun_pct'], quality_category=df['quality_category']), test_size=0.2, random_state=42, stratify=df['quality_category'])
    print("Generated synthetic data for demo.")

# Prepare data
target_regression = ['quality_score', 'cost_overrun_pct']
target_classification = 'quality_category'
X_train = train_df.drop(target_regression + [target_classification], axis=1)
X_test = test_df.drop(target_regression + [target_classification], axis=1)
y_train_reg_q, y_test_reg_q = train_df['quality_score'], test_df['quality_score']
y_train_reg_r, y_test_reg_r = train_df['cost_overrun_pct'], test_df['cost_overrun_pct']
y_train_class, y_test_class = train_df['quality_category'], test_df['quality_category']

# Encode classification target
le = LabelEncoder()
y_train_class_enc = le.fit_transform(y_train_class)
y_test_class_enc = le.transform(y_test_class)
joblib.dump(le, 'label_encoder.pkl')  # Save for inference

# Regression Models (Quality Score)
models_reg_q = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}
results_reg_q = {}
for name, model in models_reg_q.items():
    model.fit(X_train, y_train_reg_q)
    y_pred = model.predict(X_test)
    results_reg_q[name] = {
        'MAE': mean_absolute_error(y_test_reg_q, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg_q, y_pred)),
        'R2': r2_score(y_test_reg_q, y_pred)
    }
    joblib.dump(model, f'{name.lower()}_quality_reg.pkl')

# Regression Models (Cost Overrun)
models_reg_r = {name: type(model)() for name, model in models_reg_q.items()}  # Reuse
results_reg_r = {}
for name, model in models_reg_r.items():
    model.fit(X_train, y_train_reg_r)
    y_pred = model.predict(X_test)
    results_reg_r[name] = {
        'MAE': mean_absolute_error(y_test_reg_r, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg_r, y_pred)),
        'R2': r2_score(y_test_reg_r, y_pred)
    }
    joblib.dump(model, f'{name.lower()}_risk_reg.pkl')

# Classification Models
models_class = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}
results_class = {}
for name, model in models_class.items():
    model.fit(X_train, y_train_class_enc)
    y_pred = model.predict(X_test)
    results_class[name] = {
        'Accuracy': accuracy_score(y_test_class_enc, y_pred),
        'F1': f1_score(y_test_class_enc, y_pred, average='weighted')
    }
    joblib.dump(model, f'{name.lower()}_quality_class.pkl')

# Print Results
print("Regression Metrics for Quality Score:")
print("-" * 50)
for name, metrics in results_reg_q.items():
    print(f"{name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")

print("\nRegression Metrics for Cost Overrun %:")
print("-" * 50)
for name, metrics in results_reg_r.items():
    print(f"{name}: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, R2={metrics['R2']:.3f}")

print("\nClassification Metrics for Quality Category:")
print("-" * 50)
for name, metrics in results_class.items():
    print(f"{name}: Accuracy={metrics['Accuracy']:.3f}, F1={metrics['F1']:.3f}")

print("\nModels saved for inference. Use e.g., joblib.load('randomforest_quality_reg.pkl').predict(new_features)")