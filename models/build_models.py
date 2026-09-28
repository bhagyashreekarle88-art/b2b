import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(BASE), 'AADM B2B DATA SET.csv')
MODEL_DIR = os.path.join(BASE, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

df = pd.read_csv(DATA)

# ---------------- Linear Regression: Potential Deal Size ----------------
lin_num = [
    'NumberOfEmployees','AnnualRevenue_Million','LeadScore','WebsiteVisits',
    'EmailOpens','EmailClicks','SalesCalls','MeetingsAttended','WebinarsAttended',
    'EngagementScore','MarketingSpend','DiscountPercent','SalesCycleDays'
]
lin_cat = ['Industry','Region','CompanySize','LeadSource','DecisionMakerLevel','ProductInterest']
lin_X = df[lin_num + lin_cat]
lin_y = df['PotentialDealSize']
lin_X_train, lin_X_test, lin_y_train, lin_y_test = train_test_split(
    lin_X, lin_y, test_size=0.20, random_state=42
)
lin_pre = ColumnTransformer([
    ('categorical', OneHotEncoder(drop='first', handle_unknown='ignore'), lin_cat)
], remainder='passthrough')
lin_model = Pipeline([
    ('preprocessor', lin_pre),
    ('regressor', LinearRegression())
])
lin_model.fit(lin_X_train, lin_y_train)

# ---------------- Logistic Regression: Conversion ----------------
log_num = [
    'NumberOfEmployees','AnnualRevenue_Million','LeadScore','WebsiteVisits',
    'EmailOpens','EmailClicks','SalesCalls','MeetingsAttended','WebinarsAttended',
    'EngagementScore','MarketingSpend','DiscountPercent','PotentialDealSize','SalesCycleDays'
]
log_cat = ['Industry','Region','CompanySize','LeadSource','DecisionMakerLevel','ProductInterest']
log_X = df[log_num + log_cat]
log_y = df['Converted']
log_X_train, log_X_test, log_y_train, log_y_test = train_test_split(
    log_X, log_y, test_size=0.20, random_state=42, stratify=log_y
)
log_pre = ColumnTransformer([
    ('num', StandardScaler(), log_num),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), log_cat)
])
log_model = Pipeline([
    ('preprocessor', log_pre),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
log_model.fit(log_X_train, log_y_train)

# ---------------- K-Means: Opportunity Segment ----------------
cluster_features = ['AnnualRevenue_Million','LeadScore','EngagementScore','SalesCycleDays']
cluster_scaler = StandardScaler()
X_cluster_scaled = cluster_scaler.fit_transform(df[cluster_features])
kmeans = KMeans(n_clusters=2, random_state=42, n_init=20)
kmeans.fit(X_cluster_scaled)

# Preserve the dataset-derived label convention used in the analysis.
cluster_means = df.assign(Cluster=kmeans.labels_).groupby('Cluster')['AnnualRevenue_Million'].mean()
high_scale_cluster = int(cluster_means.idxmax())
low_scale_cluster = int(cluster_means.idxmin())

# Save models and deployment metadata.
joblib.dump(lin_model, os.path.join(MODEL_DIR, 'linear_potential_deal_model.joblib'))
joblib.dump(log_model, os.path.join(MODEL_DIR, 'logistic_conversion_model.joblib'))
joblib.dump(cluster_scaler, os.path.join(MODEL_DIR, 'cluster_scaler.joblib'))
joblib.dump(kmeans, os.path.join(MODEL_DIR, 'kmeans_opportunity_model.joblib'))

metadata = {
    'linear_numeric': lin_num,
    'linear_categorical': lin_cat,
    'logistic_numeric': log_num,
    'logistic_categorical': log_cat,
    'cluster_features': cluster_features,
    'high_scale_cluster': high_scale_cluster,
    'low_scale_cluster': low_scale_cluster,
    'cluster_labels': {
        str(low_scale_cluster): 'Lower-Scale Opportunity Segment',
        str(high_scale_cluster): 'High-Scale Opportunity Segment'
    },
    'model_note': 'Models retrained using the final notebook variable definitions, preprocessing and random states.'
}
joblib.dump(metadata, os.path.join(MODEL_DIR, 'metadata.joblib'))

print('Models built successfully.')
print('Linear model predictors:', len(lin_num + lin_cat))
print('Logistic model predictors:', len(log_num + log_cat))
print('High-scale cluster:', high_scale_cluster)
print('Low-scale cluster:', low_scale_cluster)
