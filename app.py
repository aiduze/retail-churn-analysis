import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Retail Churn Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    if os.path.exists('data/retail_churn_scored.csv'):
        df = pd.read_csv('data/retail_churn_scored.csv')
        return df
    elif os.path.exists('data/retail_churn_dataset.csv'):
        df = pd.read_csv('data/retail_churn_dataset.csv')
    else:
        np.random.seed(42)
        n = 6000
        df = pd.DataFrame({
            'CustomerID': [f'CUST_{str(i).zfill(6)}' for i in range(1, n+1)],
            'Churn': np.random.binomial(1, 0.18, n),
            'Tenure': np.random.exponential(12, n).clip(1, 60).astype(int),
            'PreferredLoginDevice': np.random.choice(['Mobile', 'Computer', 'Tablet'], n, p=[0.55, 0.35, 0.10]),
            'CityTier': np.random.choice([1, 2, 3], n, p=[0.30, 0.45, 0.25]),
            'WarehouseToHome': np.random.gamma(3, 5, n).clip(1, 50).round(1),
            'PreferredPaymentMode': np.random.choice(['Credit Card', 'UPI', 'E-wallet', 'Cash on Delivery', 'Debit Card'], n, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
            'Gender': np.random.choice(['Male', 'Female'], n, p=[0.52, 0.48]),
            'HourSpendOnApp': np.random.gamma(2, 1.5, n).clip(0.5, 10).round(1),
            'NumberOfDeviceRegistered': np.random.choice(range(1, 7), n, p=[0.15, 0.25, 0.25, 0.18, 0.12, 0.05]),
            'PreferedOrderCat': np.random.choice(['Fashion', 'Electronics', 'Groceries', 'Mobile', 'Home & Kitchen'], n, p=[0.28, 0.22, 0.18, 0.17, 0.15]),
            'SatisfactionScore': np.random.choice(range(1, 11), n, p=[0.05, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.12, 0.10]),
            'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced'], n, p=[0.35, 0.50, 0.15]),
            'NumberOfAddress': np.random.choice(range(1, 6), n, p=[0.30, 0.35, 0.20, 0.10, 0.05]),
            'Complain': np.random.binomial(1, 0.15, n),
            'OrderAmountHikeFromlastYear': np.random.normal(15, 12, n).clip(-20, 60).round(1),
            'CouponUsed': np.random.poisson(2, n).clip(0, 10),
            'OrderCount': np.random.poisson(3, n).clip(0, 15),
            'DaySinceLastOrder': np.random.exponential(15, n).clip(1, 90).astype(int),
            'CashbackAmount': np.random.exponential(50, n).clip(0, 300).round(2),
        })
        for col in ['SatisfactionScore', 'CashbackAmount', 'OrderAmountHikeFromlastYear']:
            df.loc[np.random.choice(df.index, size=int(0.025*len(df)), replace=False), col] = np.nan
        df['SatisfactionScore'].fillna(df['SatisfactionScore'].median(), inplace=True)
        df['OrderAmountHikeFromlastYear'].fillna(df['OrderAmountHikeFromlastYear'].median(), inplace=True)
        df['CashbackAmount'].fillna(df['CashbackAmount'].median(), inplace=True)

    df['EngagementScore'] = (df['HourSpendOnApp'] * 0.3 + df['OrderCount'] * 0.3 + df['CouponUsed'] * 0.2 + (df['CashbackAmount'] / 50) * 0.2).round(2)
    df['RiskScore'] = ((10 - df['SatisfactionScore']) * 0.25 + (df['DaySinceLastOrder'] / 30) * 0.25 + (df['Complain'] * 5) * 0.20 + (df['Tenure'] < 6).astype(int) * 3 * 0.15 + (df['OrderCount'] == 0).astype(int) * 5 * 0.15).round(2)
    df['ValueTier'] = pd.qcut(df['OrderAmountHikeFromlastYear'].clip(-20, 60), q=3, labels=['Low', 'Medium', 'High'])
    df['RecencyScore'] = pd.cut(df['DaySinceLastOrder'], bins=[0, 7, 14, 30, 90], labels=['Very Recent', 'Recent', 'Moderate', 'At Risk'])

    cat_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'PreferedOrderCat', 'MaritalStatus']
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col + '_enc'] = LabelEncoder().fit_transform(df_enc[col])

    model_cols = ['Tenure', 'CityTier', 'WarehouseToHome', 'HourSpendOnApp',
                  'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                  'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
                  'DaySinceLastOrder', 'CashbackAmount', 'EngagementScore', 'RiskScore']
    for col in cat_cols:
        model_cols.append(col + '_enc')

    X = df_enc[model_cols]
    y = df_enc['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    df['ChurnProbability'] = rf.predict_proba(X)[:, 1]

    df['RiskSegment'] = pd.cut(df['ChurnProbability'], bins=[0, 0.2, 0.5, 0.8, 1.0], labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])

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
    return df


df = load_data()

# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.title("🛒 Filters")

risk_filter = st.sidebar.multiselect(
    "Risk segment",
    options=df['RiskSegment'].cat.categories.tolist(),
    default=df['RiskSegment'].cat.categories.tolist()
)

category_filter = st.sidebar.multiselect(
    "Product category",
    options=sorted(df['PreferedOrderCat'].unique()),
    default=sorted(df['PreferedOrderCat'].unique())
)

payment_filter = st.sidebar.multiselect(
    "Payment method",
    options=sorted(df['PreferredPaymentMode'].unique()),
    default=sorted(df['PreferredPaymentMode'].unique())
)

city_filter = st.sidebar.multiselect(
    "City tier",
    options=sorted(df['CityTier'].unique()),
    default=sorted(df['CityTier'].unique())
)

satisfaction_range = st.sidebar.slider(
    "Satisfaction score",
    min_value=int(df['SatisfactionScore'].min()),
    max_value=int(df['SatisfactionScore'].max()),
    value=(1, 10)
)

tenure_range = st.sidebar.slider(
    "Tenure (months)",
    min_value=int(df['Tenure'].min()),
    max_value=int(df['Tenure'].max()),
    value=(1, 60)
)

filtered = df[
    (df['RiskSegment'].isin(risk_filter)) &
    (df['PreferedOrderCat'].isin(category_filter)) &
    (df['PreferredPaymentMode'].isin(payment_filter)) &
    (df['CityTier'].isin(city_filter)) &
    (df['SatisfactionScore'] >= satisfaction_range[0]) &
    (df['SatisfactionScore'] <= satisfaction_range[1]) &
    (df['Tenure'] >= tenure_range[0]) &
    (df['Tenure'] <= tenure_range[1])
]

# ============================================================
# HEADER
# ============================================================
st.title("Retail Customer Churn & Retention Dashboard")
st.markdown("Monitor at-risk customers, segment by behavior, and prioritize retention actions.")

# ============================================================
# KPI ROW
# ============================================================
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Total Customers", f"{len(filtered):,}")
col2.metric("Churned", f"{filtered['Churn'].sum():,}", f"{filtered['Churn'].mean()*100:.1f}%")
col3.metric("Avg Churn Risk", f"{filtered['ChurnProbability'].mean()*100:.1f}%")
col4.metric("Avg Satisfaction", f"{filtered['SatisfactionScore'].mean():.1f}")
col5.metric("Critical Risk", f"{(filtered['RiskSegment'] == 'Critical Risk').sum():,}")

st.divider()

# ============================================================
# CHARTS ROW 1
# ============================================================
left, right = st.columns(2)

with left:
    st.subheader("Risk Segment Distribution")
    risk_counts = filtered['RiskSegment'].value_counts().reindex(['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'], fill_value=0)
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(risk_counts.index, risk_counts.values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Customers")
    ax.set_ylim(0, max(risk_counts.values) * 1.15)
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(risk_counts.values)*0.02,
                f"{val:,}", ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with right:
    st.subheader("Churn Rate by Key Driver")
    driver = st.selectbox("Select driver", ['SatisfactionScore', 'Tenure', 'DaySinceLastOrder', 'PreferedOrderCat', 'PreferredPaymentMode'], key="driver1")
    fig, ax = plt.subplots(figsize=(6, 4))
    if driver in ['SatisfactionScore', 'Tenure', 'DaySinceLastOrder']:
        if driver == 'SatisfactionScore':
            bins = range(1, 12)
            labels = list(range(1, 11))
        elif driver == 'Tenure':
            bins = [0, 6, 12, 24, 60]
            labels = ['0-6m', '7-12m', '13-24m', '25m+']
        else:
            bins = [0, 7, 14, 30, 90]
            labels = ['0-7d', '8-14d', '15-30d', '31d+']
        filtered['temp_bin'] = pd.cut(filtered[driver], bins=bins, labels=labels)
        churn_by = filtered.groupby('temp_bin')['Churn'].mean() * 100
    else:
        churn_by = filtered.groupby(driver)['Churn'].mean() * 100
        churn_by = churn_by.sort_values(ascending=False)

    bars = ax.bar(churn_by.index.astype(str), churn_by.values, color='#e74c3c', alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, max(churn_by.values) * 1.2 if len(churn_by) > 0 else 10)
    ax.tick_params(axis='x', rotation=30)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

# ============================================================
# CHARTS ROW 2
# ============================================================
left2, right2 = st.columns(2)

with left2:
    st.subheader("Satisfaction vs Churn Risk")
    fig, ax = plt.subplots(figsize=(6, 4))
    retained = filtered[filtered['Churn'] == 0]
    churned = filtered[filtered['Churn'] == 1]
    ax.scatter(retained['SatisfactionScore'], retained['ChurnProbability'], alpha=0.3, s=15, color='#2ecc71', label='Retained')
    ax.scatter(churned['SatisfactionScore'], churned['ChurnProbability'], alpha=0.5, s=20, color='#e74c3c', label='Churned')
    ax.set_xlabel("Satisfaction Score")
    ax.set_ylabel("Churn Probability")
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    st.pyplot(fig)

with right2:
    st.subheader("Retention Campaign ROI Simulator")
    intervention_rate = st.slider("Intervention rate (% of customers)", 5, 50, 20)
    threshold = filtered['ChurnProbability'].quantile(1 - intervention_rate / 100)
    targeted = filtered[filtered['ChurnProbability'] >= threshold]
    churners_caught = targeted['Churn'].sum()
    total_churners = filtered['Churn'].sum()
    capture_rate = (churners_caught / total_churners * 100) if total_churners > 0 else 0

    fig, ax = plt.subplots(figsize=(6, 4))
    categories = ['Targeted', 'Not Targeted']
    values = [len(targeted), len(filtered) - len(targeted)]
    colors_roi = ['#e74c3c', '#bdc3c7']
    wedges, texts, autotexts = ax.pie(values, labels=categories, autopct='%1.1f%%', colors=colors_roi, startangle=90)
    ax.set_title(f"Targeting top {intervention_rate}% captures {capture_rate:.0f}% of churners")
    st.pyplot(fig)

st.divider()

# ============================================================
# CUSTOMER TABLE
# ============================================================
st.subheader("Customer Risk Register")

table_cols = ['CustomerID', 'RiskSegment', 'ChurnProbability', 'SatisfactionScore',
              'Tenure', 'DaySinceLastOrder', 'Complain', 'PreferedOrderCat',
              'PreferredPaymentMode', 'ActionSegment']

display_df = filtered[table_cols].copy()
display_df['ChurnProbability'] = (display_df['ChurnProbability'] * 100).round(1).astype(str) + '%'
display_df = display_df.sort_values('ChurnProbability', ascending=False)

st.dataframe(display_df, use_container_width=True, hide_index=True)

csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download filtered customer list",
    data=csv,
    file_name=f"churn_customers_{len(filtered)}.csv",
    mime="text/csv"
)

st.divider()

# ============================================================
# ACTIONABLE INSIGHTS
# ============================================================
st.subheader("Recommended Actions")

action_summary = filtered.groupby('ActionSegment').agg({
    'CustomerID': 'count',
    'ChurnProbability': 'mean'
}).round(3)
action_summary.columns = ['Customers', 'Avg Risk']
action_summary = action_summary.sort_values('Customers', ascending=False)

for segment, row in action_summary.iterrows():
    with st.expander(f"{segment} — {int(row['Customers'])} customers (avg risk: {row['Avg Risk']*100:.1f}%)"):
        if 'Critical' in segment:
            st.markdown('- Immediate personal outreach within 24 hours')
            st.markdown('- Offer retention discount (15-20%) or free shipping for 3 months')
            st.markdown('- Assign dedicated account manager')
        elif 'Complaint' in segment:
            st.markdown('- Escalate to senior CS for resolution')
            st.markdown('- Compensate with store credit or refund')
            st.markdown('- Follow up within 48 hours to confirm satisfaction')
        elif 'Onboarding' in segment:
            st.markdown('- Redesign first-30-day journey: welcome series, tutorial prompts')
            st.markdown('- Personalized product recommendations based on first purchase')
            st.markdown('- Check-in call at day 14')
        elif 'Retention Campaign' in segment:
            st.markdown('- Launch targeted email/SMS campaign with limited-time offer')
            st.markdown('- Bundle recommendations to increase basket size')
            st.markdown('- Loyalty points multiplier for next purchase')
        elif 'Service Recovery' in segment:
            st.markdown('- Proactive outreach before they churn')
            st.markdown('- Service quality audit on recent interactions')
            st.markdown('- Offer upgrade or complimentary service')
        elif 'Dormant' in segment:
            st.markdown('- Win-back email with We miss you + exclusive discount')
            st.markdown('- Retargeting ads on social platforms')
            st.markdown('- Survey to understand why they stopped ordering')
        else:
            st.markdown('- VIP loyalty program enrollment')
            st.markdown('- Referral incentives (give $20, get $20)')
            st.markdown('- Early access to new products/sales')

st.caption("Built with Streamlit — Data is synthetic for demonstration purposes")
