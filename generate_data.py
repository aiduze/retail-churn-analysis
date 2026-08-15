"""
generate_data.py
================
Generates the synthetic retail churn dataset used in the analysis.
Run this first to create data/retail_churn_dataset.csv

Reproducibility: fixed random seed (42) ensures identical output every time.
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

N_CUSTOMERS = 6000

# --- Customer Demographics ---
customer_id = [f'CUST_{str(i).zfill(6)}' for i in range(1, N_CUSTOMERS + 1)]
gender = np.random.choice(['Male', 'Female'], N_CUSTOMERS, p=[0.52, 0.48])
marital_status = np.random.choice(['Single', 'Married', 'Divorced'], N_CUSTOMERS, p=[0.35, 0.50, 0.15])
city_tier = np.random.choice([1, 2, 3], N_CUSTOMERS, p=[0.30, 0.45, 0.25])

# --- Behavioral Features ---
tenure = np.random.exponential(scale=12, size=N_CUSTOMERS).clip(1, 60).astype(int)

satisfaction = np.random.choice(
    range(1, 11), N_CUSTOMERS,
    p=[0.05, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.12, 0.10]
)

hour_spend_on_app = np.random.gamma(shape=2, scale=1.5, size=N_CUSTOMERS).clip(0.5, 10).round(1)
num_devices = np.random.choice(range(1, 7), N_CUSTOMERS, p=[0.15, 0.25, 0.25, 0.18, 0.12, 0.05])
num_addresses = np.random.choice(range(1, 6), N_CUSTOMERS, p=[0.30, 0.35, 0.20, 0.10, 0.05])
warehouse_to_home = np.random.gamma(shape=3, scale=5, size=N_CUSTOMERS).clip(1, 50).round(1)

preferred_login = np.random.choice(['Mobile', 'Computer', 'Tablet'], N_CUSTOMERS, p=[0.55, 0.35, 0.10])
preferred_payment = np.random.choice(
    ['Credit Card', 'UPI', 'E-wallet', 'Cash on Delivery', 'Debit Card'],
    N_CUSTOMERS, p=[0.30, 0.25, 0.20, 0.15, 0.10]
)
preferred_category = np.random.choice(
    ['Fashion', 'Electronics', 'Groceries', 'Mobile', 'Home & Kitchen'],
    N_CUSTOMERS, p=[0.28, 0.22, 0.18, 0.17, 0.15]
)

# --- Transactional Features ---
order_count = np.random.poisson(lam=3, size=N_CUSTOMERS).clip(0, 15)
days_since_last = np.random.exponential(scale=15, size=N_CUSTOMERS).clip(1, 90).astype(int)
order_hike = np.random.normal(loc=15, scale=12, size=N_CUSTOMERS).clip(-20, 60).round(1)
coupon_used = np.random.poisson(lam=2, size=N_CUSTOMERS).clip(0, 10)
cashback = np.random.exponential(scale=50, size=N_CUSTOMERS).clip(0, 300).round(2)

# Complaints: inversely correlated with satisfaction
complain_prob = 1 - (satisfaction / 10) ** 2
complain = np.random.binomial(1, complain_prob * 0.4)

# --- CHURN GENERATION ---
# Composite score from realistic business drivers
churn_score = (
    0.25 * (1 - satisfaction / 10) +
    0.20 * (days_since_last / 90) +
    0.15 * (tenure < 6).astype(float) +
    0.15 * complain +
    0.10 * (order_count == 0).astype(float) +
    0.08 * (order_hike < 0).astype(float) +
    0.05 * (cashback < 20).astype(float) +
    0.02 * (num_devices > 4).astype(float)
)

churn_score += np.random.normal(0, 0.05, N_CUSTOMERS)
churn_score = np.clip(churn_score, 0, 1)

# Convert to binary (~18% churn rate)
churn = (churn_score > np.percentile(churn_score, 82)).astype(int)

# --- Assemble DataFrame ---
df = pd.DataFrame({
    'CustomerID': customer_id,
    'Churn': churn,
    'Tenure': tenure,
    'PreferredLoginDevice': preferred_login,
    'CityTier': city_tier,
    'WarehouseToHome': warehouse_to_home,
    'PreferredPaymentMode': preferred_payment,
    'Gender': gender,
    'HourSpendOnApp': hour_spend_on_app,
    'NumberOfDeviceRegistered': num_devices,
    'PreferedOrderCat': preferred_category,
    'SatisfactionScore': satisfaction,
    'MaritalStatus': marital_status,
    'NumberOfAddress': num_addresses,
    'Complain': complain,
    'OrderAmountHikeFromlastYear': order_hike,
    'CouponUsed': coupon_used,
    'OrderCount': order_count,
    'DaySinceLastOrder': days_since_last,
    'CashbackAmount': cashback
})

# Introduce realistic missing values (2.5%)
for col in ['SatisfactionScore', 'CashbackAmount', 'OrderAmountHikeFromlastYear']:
    missing_idx = np.random.choice(df.index, size=int(0.025 * len(df)), replace=False)
    df.loc[missing_idx, col] = np.nan

# Save
os.makedirs('data', exist_ok=True)
df.to_csv('data/retail_churn_dataset.csv', index=False)

print(f"Dataset generated: {N_CUSTOMERS} customers x {df.shape[1]} features")
print(f"Churn rate: {df['Churn'].mean()*100:.1f}%")
print(f"Saved to: data/retail_churn_dataset.csv")
