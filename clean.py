import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('synthetic_dpr_metadata_outcomes.csv')
print("Original data shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())

# Step 0: Early drop 'id' if present
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)
    print("Dropped 'id' column.")

# Step 1: Separate features and targets (to exclude from cols)
target_cols = ['quality_score', 'quality_category', 'cost_overrun_pct']
feature_cols = [col for col in df.columns if col not in target_cols]
X_full = df[feature_cols].copy()
y_quality = df['quality_score']
y_risk = df['cost_overrun_pct']
y_category = df['quality_category']  # For stratify

# Convert bool 'delay' to int (0/1) for numeric handling
if 'delay' in X_full.columns:
    X_full['delay'] = X_full['delay'].astype(int)

print("\nFeatures shape after separation:", X_full.shape)

# Step 2: Missing Value Imputation on features only
# Identify numeric and categorical columns ON FEATURES
numeric_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_full.select_dtypes(include=['object']).columns.tolist()

# For numerics: Use median if skewed (IQR-based), else mean; mode fallback
for col in numeric_cols:
    if X_full[col].isnull().sum() > 0:
        Q1 = X_full[col].quantile(0.25)
        Q3 = X_full[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            # Simple skew check: median vs mean diff relative to std
            skew_ratio = abs((X_full[col].median() - X_full[col].mean())) / X_full[col].std() if X_full[col].std() > 0 else 0
            if skew_ratio > 0.5:  # Threshold for skew
                X_full[col].fillna(X_full[col].median(), inplace=True)  # Median for skewed
            else:
                X_full[col].fillna(X_full[col].mean(), inplace=True)  # Mean for symmetric
        else:
            mode_val = X_full[col].mode()[0] if len(X_full[col].mode()) > 0 else 0
            X_full[col].fillna(mode_val, inplace=True)  # Mode fallback

# For categoricals: Mode
for col in categorical_cols:
    if X_full[col].isnull().sum() > 0:
        mode_val = X_full[col].mode()[0] if len(X_full[col].mode()) > 0 else 'Unknown'
        X_full[col].fillna(mode_val, inplace=True)

print("\nAfter imputation - Missing values in features:\n", X_full.isnull().sum().sum())  # Should be 0

# Step 3: Outlier Handling (Clip to 1.5*IQR) on numerics
for col in numeric_cols:
    Q1 = X_full[col].quantile(0.25)
    Q3 = X_full[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    X_full[col] = np.clip(X_full[col], lower_bound, upper_bound)

# Step 4: Normalization & Encoding Pipeline (now on clean features only)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),  # Normalize numerics (mean=0, std=1)
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)  # One-hot cats
    ]
)

# Fit and transform features
X_transformed = preprocessor.fit_transform(X_full)

# Get feature names post-transform
num_features = len(numeric_cols)
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
all_feature_names = numeric_cols + list(cat_features)  # Note: numeric_cols already names

# Create cleaned DF: scaled features + targets
X_df = pd.DataFrame(X_transformed, columns=all_feature_names)
cleaned_df = pd.concat([X_df, df[target_cols]], axis=1)

# Step 5: Train-Test Split (Stratified on quality_category)
X_train, X_test, y_train_quality, y_test_quality, y_train_risk, y_test_risk, y_train_category, y_test_category = train_test_split(
    cleaned_df.drop(target_cols, axis=1),
    cleaned_df['quality_score'],
    cleaned_df['cost_overrun_pct'],
    cleaned_df['quality_category'],
    test_size=0.2, random_state=42, stratify=cleaned_df['quality_category']  # Balance classes
)

# Save split datasets (include category for full ref)
train_df = pd.concat([
    X_train, 
    pd.DataFrame({
        'quality_score': y_train_quality, 
        'cost_overrun_pct': y_train_risk,
        'quality_category': y_train_category
    })
], axis=1)
test_df = pd.concat([
    X_test, 
    pd.DataFrame({
        'quality_score': y_test_quality, 
        'cost_overrun_pct': y_test_risk,
        'quality_category': y_test_category
    })
], axis=1)
train_df.to_csv('cleaned_dpr_train.csv', index=False)
test_df.to_csv('cleaned_dpr_test.csv', index=False)

# Save preprocessor for inference
joblib.dump(preprocessor, 'preprocessor.pkl')

print("\nCleaned data stats:")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("\nSample scaled numeric (funding_amount_cr mean/std in train):")
if 'funding_amount_cr' in X_train.columns:
    print(f"Train mean: {X_train['funding_amount_cr'].mean():.3f}, std: {X_train['funding_amount_cr'].std():.3f}")  # ~0, ~1
    print(f"Test mean: {X_test['funding_amount_cr'].mean():.3f}, std: {X_test['funding_amount_cr'].std():.3f}")

print("\nFinal check - Any NaNs in train?", train_df.isnull().sum().sum())
print("Quality category distribution in train:\n", train_df['quality_category'].value_counts(normalize=True))