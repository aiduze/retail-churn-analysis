"""
churn_analysis.py
=================
End-to-end customer churn analysis pipeline.

Run after generate_data.py to produce all outputs.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve,
    confusion_matrix, precision_recall_curve, average_precision_score
)

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
df = pd.read_csv('data/retail_churn_dataset.csv')

# Handle missing values
df['SatisfactionScore'].fillna(df['SatisfactionScore'].median(), inplace=True)
df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median(), inplace=True)
df['CashbackAmount'].fillna(df['CashbackAmount'].median(), inplace=True)

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
df['EngagementScore'] = (
    df['HourSpendOnApp'] * 0.3 +
    df['OrderCount'] * 0.3 +
    df['CouponUsed'] * 0.2 +
    (df['CashbackAmount'] / 50) * 0.2
).round(2)

df['RiskScore'] = (
    (10 - df['SatisfactionScore']) * 0.25 +
    (df['DaySinceLastOrder'] / 30) * 0.25 +
    (df['Complain'] * 5) * 0.20 +
    (df['Tenure'] < 6).astype(int) * 3 * 0.15 +
    (df['OrderCount'] == 0).astype(int) * 5 * 0.15
).round(2)

df['ValueTier'] = pd.qcut(
    df['OrderAmountHikeFromlastYear'].clip(-20, 60),
    q=3, labels=['Low', 'Medium', 'High']
)

df['RecencyScore'] = pd.cut(
    df['DaySinceLastOrder'],
    bins=[0, 7, 14, 30, 90],
    labels=['Very Recent', 'Recent', 'Moderate', 'At Risk']
)

# ============================================================
# 3. PREPARE FOR MODELING
# ============================================================
categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender',
                    'PreferedOrderCat', 'MaritalStatus']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col])

model_cols = ['Churn', 'Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
              'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
              'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
              'DaySinceLastOrder', 'CashbackAmount', 'EngagementScore', 'RiskScore']

for col in categorical_cols:
    model_cols.append(col + '_enc')

df_model = df[model_cols].copy()

# ============================================================
# 4. MODELING
# ============================================================
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42),
}

results = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    results[name] = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'auc': auc,
        'ap': ap
    }

    print(f"\n{name}")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  Avg Precision: {ap:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Retained', 'Churned']))

# ============================================================
# 5. SEGMENTATION
# ============================================================
rf_model = results['Random Forest']['model']
df['ChurnProbability'] = rf_model.predict_proba(X)[:, 1]

df['RiskSegment'] = pd.cut(
    df['ChurnProbability'],
    bins=[0, 0.2, 0.5, 0.8, 1.0],
    labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
)

def segment_customers(row):
    if row['ChurnProbability'] > 0.8:
        return 'Critical: Immediate Intervention'
    elif row['ChurnProbability'] > 0.5 and row['Complain'] == 1:
        return 'High Risk + Complaint: Escalate'
    elif row['ChurnProbability'] > 0.5 and row['Tenure'] < 6:
        return 'High Risk + New: Onboarding Fix'
    elif row['ChurnProbability'] > 0.5:
        return 'High Risk: Retention Campaign'
    elif row['SatisfactionScore'] <= 4:
        return 'Low Satisfaction: Service Recovery'
    elif row['DaySinceLastOrder'] > 30:
        return 'Dormant: Re-engagement'
    else:
        return 'Stable: Loyalty Program'

df['ActionSegment'] = df.apply(segment_customers, axis=1)

# Save scored dataset
df.to_csv('data/retail_churn_scored.csv', index=False)
print("\nScored dataset saved to: data/retail_churn_scored.csv")

# ============================================================
# 6. EXPORT VISUALIZATIONS
# ============================================================
os.makedirs('outputs', exist_ok=True)

# EDA Chart
fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('Customer Churn Analysis: Key Drivers', fontsize=16, fontweight='bold', y=1.02)

ax = axes[0, 0]
sat_churn = df.groupby('SatisfactionScore')['Churn'].mean() * 100
sat_churn.plot(kind='bar', ax=ax, color='#e74c3c', alpha=0.8)
ax.set_title('Churn Rate by Satisfaction Score', fontweight='bold')
ax.set_xlabel('Satisfaction Score (1-10)')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 1]
tenure_bins = pd.cut(df['Tenure'], bins=[0, 6, 12, 24, 60], labels=['0-6m', '7-12m', '13-24m', '25m+'])
tenure_churn = df.groupby(tenure_bins)['Churn'].mean() * 100
tenure_churn.plot(kind='bar', ax=ax, color='#3498db', alpha=0.8)
ax.set_title('Churn Rate by Tenure', fontweight='bold')
ax.set_xlabel('Tenure')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[0, 2]
days_bins = pd.cut(df['DaySinceLastOrder'], bins=[0, 7, 14, 30, 90], labels=['0-7d', '8-14d', '15-30d', '31d+'])
days_churn = df.groupby(days_bins)['Churn'].mean() * 100
days_churn.plot(kind='bar', ax=ax, color='#f39c12', alpha=0.8)
ax.set_title('Churn Rate by Days Since Last Order', fontweight='bold')
ax.set_xlabel('Days Since Last Order')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 0]
pay_churn = df.groupby('PreferredPaymentMode')['Churn'].mean() * 100
pay_churn.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#9b59b6', alpha=0.8)
ax.set_title('Churn Rate by Payment Method', fontweight='bold')
ax.set_xlabel('Payment Method')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
city_churn = df.groupby('CityTier')['Churn'].mean() * 100
city_churn.plot(kind='bar', ax=ax, color='#1abc9c', alpha=0.8)
ax.set_title('Churn Rate by City Tier', fontweight='bold')
ax.set_xlabel('City Tier')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 2]
cat_churn = df.groupby('PreferedOrderCat')['Churn'].mean() * 100
cat_churn.sort_values(ascending=False).plot(kind='bar', ax=ax, color='#e67e22', alpha=0.8)
ax.set_title('Churn Rate by Product Category', fontweight='bold')
ax.set_xlabel('Preferred Category')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)

ax = axes[2, 0]
comp_churn = df.groupby('Complain')['Churn'].mean() * 100
comp_churn.plot(kind='bar', ax=ax, color='#c0392b', alpha=0.8)
ax.set_title('Churn Rate by Complaint Status', fontweight='bold')
ax.set_xlabel('Complaint Filed (0=No, 1=Yes)')
ax.set_ylabel('Churn Rate (%)')
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', alpha=0.3)

ax = axes[2, 1]
df[df['Churn']==0]['Tenure'].hist(bins=30, alpha=0.6, label='Retained', ax=ax, color='#2ecc71')
df[df['Churn']==1]['Tenure'].hist(bins=30, alpha=0.6, label='Churned', ax=ax, color='#e74c3c')
ax.set_title('Tenure Distribution by Churn', fontweight='bold')
ax.set_xlabel('Tenure (months)')
ax.set_ylabel('Count')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[2, 2]
numeric_cols = ['Tenure', 'SatisfactionScore', 'HourSpendOnApp', 'NumberOfDeviceRegistered',
                'NumberOfAddress', 'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed',
                'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'Churn']
corr = df[numeric_cols].corr()
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(numeric_cols)))
ax.set_yticks(range(len(numeric_cols)))
ax.set_xticklabels([c.replace('OrderAmountHikeFromlastYear', 'OrderHike') for c in numeric_cols], rotation=45, ha='right', fontsize=8)
ax.set_yticklabels([c.replace('OrderAmountHikeFromlastYear', 'OrderHike') for c in numeric_cols], fontsize=8)
ax.set_title('Feature Correlation Matrix', fontweight='bold')
for i in range(len(numeric_cols)):
    for j in range(len(numeric_cols)):
        ax.text(j, i, f'{corr.iloc[i,j]:.1f}', ha='center', va='center', fontsize=7,
                color='white' if abs(corr.iloc[i,j]) > 0.5 else 'black')

plt.tight_layout()
plt.savefig('outputs/01_eda_churn_drivers.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: outputs/01_eda_churn_drivers.png")

# Model Performance Chart
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Model Performance: Churn Prediction', fontsize=16, fontweight='bold', y=1.02)

ax = axes[0, 0]
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

ax = axes[0, 1]
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['y_proba'])
    ax.plot(recall, precision, label=f"{name} (AP={res['ap']:.3f})", linewidth=2)
ax.axhline(y=y_test.mean(), color='k', linestyle='--', alpha=0.5, label=f'Baseline ({y_test.mean():.2f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves', fontweight='bold')
ax.legend(loc='lower left')
ax.grid(alpha=0.3)

ax = axes[0, 2]
cm = confusion_matrix(y_test, results['Random Forest']['y_pred'])
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Retained', 'Churned'])
ax.set_yticklabels(['Retained', 'Churned'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix (Random Forest)', fontweight='bold')
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold',
                color='white' if cm[i,j] > cm.max()/2 else 'black')

ax = axes[1, 0]
rf_model = results['Random Forest']['model']
importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
colors = ['#e74c3c' if 'RiskScore' in i or 'Satisfaction' in i or 'DaySince' in i or 'Complain' in i or 'Tenure' in i
          else '#3498db' for i in importance.index]
importance.tail(15).plot(kind='barh', ax=ax, color=colors[-15:])
ax.set_title('Top 15 Feature Importance (Random Forest)', fontweight='bold')
ax.set_xlabel('Importance')
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 1]
lr_model = results['Logistic Regression']['model']
coef = pd.Series(lr_model.coef_[0], index=X.columns)
coef_sorted = coef.reindex(coef.abs().sort_values(ascending=True).index)
colors_lr = ['#e74c3c' if c > 0 else '#2ecc71' for c in coef_sorted.tail(15)]
coef_sorted.tail(15).plot(kind='barh', ax=ax, color=colors_lr)
ax.set_title('Top 15 Coefficients (Logistic Regression)', fontweight='bold')
ax.set_xlabel('Coefficient (log-odds)')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 2]
for name, res in results.items():
    ax.hist(res['y_proba'][y_test==0], bins=30, alpha=0.5, label=f'{name} - Retained', density=True)
    ax.hist(res['y_proba'][y_test==1], bins=30, alpha=0.5, label=f'{name} - Churned', density=True)
ax.set_xlabel('Predicted Churn Probability')
ax.set_ylabel('Density')
ax.set_title('Churn Probability Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/02_model_performance.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: outputs/02_model_performance.png")

# Feature Impact Chart
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Feature Impact on Churn Probability', fontsize=16, fontweight='bold', y=1.02)

top_features = ['RiskScore', 'SatisfactionScore', 'Tenure', 'DaySinceLastOrder', 'Complain', 'OrderAmountHikeFromlastYear']

for idx, feat in enumerate(top_features):
    ax = axes[idx // 3, idx % 3]
    if feat in ['Tenure', 'DaySinceLastOrder', 'OrderAmountHikeFromlastYear']:
        df['temp_bin'] = pd.qcut(df[feat], q=10, duplicates='drop')
    else:
        df['temp_bin'] = df[feat]
    grouped = df.groupby('temp_bin').agg({'Churn': 'mean', feat: 'mean'}).reset_index()
    ax.plot(grouped[feat], grouped['Churn'] * 100, marker='o', linewidth=2, markersize=6, color='#e74c3c')
    ax.set_xlabel(feat)
    ax.set_ylabel('Churn Rate (%)')
    ax.set_title(f'Churn Rate vs {feat}', fontweight='bold')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/03_feature_impact.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: outputs/03_feature_impact.png")

# Segmentation Strategy Chart
fig, axes = plt.subplots(2, 3, figsize=(17, 11))
fig.suptitle('Customer Segmentation & Retention Strategy', fontsize=16, fontweight='bold', y=1.02)

ax = axes[0, 0]
risk_counts = df['RiskSegment'].value_counts()
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
ax.set_title('Customer Risk Distribution', fontweight='bold')

ax = axes[0, 1]
ax.hist(df[df['Churn']==0]['ChurnProbability'], bins=40, alpha=0.6, label='Retained', color='#2ecc71', density=True)
ax.hist(df[df['Churn']==1]['ChurnProbability'], bins=40, alpha=0.6, label='Churned', color='#e74c3c', density=True)
ax.set_xlabel('Predicted Churn Probability')
ax.set_ylabel('Density')
ax.set_title('Churn Probability Distribution', fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 2]
action_counts = df['ActionSegment'].value_counts()
action_colors = ['#e74c3c', '#e67e22', '#f39c12', '#9b59b6', '#3498db', '#1abc9c', '#2ecc71']
action_counts.plot(kind='barh', ax=ax, color=action_colors[:len(action_counts)])
ax.set_title('Actionable Segments', fontweight='bold')
ax.set_xlabel('Number of Customers')
ax.grid(axis='x', alpha=0.3)

ax = axes[1, 0]
risk_actual = df.groupby('RiskSegment')['Churn'].mean() * 100
risk_predicted = df.groupby('RiskSegment')['ChurnProbability'].mean() * 100
x_pos = range(len(risk_actual))
width = 0.35
ax.bar([p - width/2 for p in x_pos], risk_actual, width, label='Actual Churn %', color='#e74c3c', alpha=0.8)
ax.bar([p + width/2 for p in x_pos], risk_predicted, width, label='Predicted Risk %', color='#3498db', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels(risk_actual.index, rotation=15, ha='right')
ax.set_ylabel('Percentage (%)')
ax.set_title('Model Calibration: Actual vs Predicted', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

ax = axes[1, 1]
scatter = ax.scatter(df['SatisfactionScore'], df['ChurnProbability'],
                     c=df['Churn'], cmap='RdYlGn_r', alpha=0.4, s=20)
ax.set_xlabel('Satisfaction Score')
ax.set_ylabel('Churn Probability')
ax.set_title('Satisfaction vs Churn Risk', fontweight='bold')
ax.grid(alpha=0.3)

ax = axes[1, 2]
intervention_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
customers_at_risk = []
churners_caught = []
for rate in intervention_rates:
    threshold = df['ChurnProbability'].quantile(1 - rate)
    targeted = df[df['ChurnProbability'] >= threshold]
    customers_at_risk.append(len(targeted))
    churners_caught.append(targeted['Churn'].sum())

ax2 = ax.twinx()
line1 = ax.plot([r*100 for r in intervention_rates], customers_at_risk, 'o-', color='#3498db', linewidth=2, label='Customers Targeted')
line2 = ax2.plot([r*100 for r in intervention_rates], [c/df['Churn'].sum()*100 for c in churners_caught], 's-', color='#e74c3c', linewidth=2, label='% Churners Caught')
ax.set_xlabel('Intervention Rate (%)')
ax.set_ylabel('Customers Targeted', color='#3498db')
ax2.set_ylabel('% of Churners Caught', color='#e74c3c')
ax.set_title('Retention Campaign ROI Simulation', fontweight='bold')
ax.grid(alpha=0.3)
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, loc='center right')

plt.tight_layout()
plt.savefig('outputs/04_segmentation_strategy.png', bbox_inches='tight', facecolor='white')
plt.close()
print("Saved: outputs/04_segmentation_strategy.png")

print("\n=== ALL OUTPUTS GENERATED ===")
print("Check outputs/ directory for all visualizations")
print("Check data/retail_churn_scored.csv for scored customer list")
