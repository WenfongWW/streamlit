import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('historical_data.csv')
    data['created_at'] = pd.to_datetime(data['created_at'])
    data['actual_delivery_time'] = pd.to_datetime(data['actual_delivery_time'])
    data['total_delivery_duration'] = (data['actual_delivery_time'] - data['created_at']).dt.total_seconds()
    return data.copy()  # Return a copy to avoid mutation

# Preprocess the data
def preprocess_data(data):
    # Feature engineering steps as defined previously
    data['avg_item_price'] = data['subtotal'] / data['total_items']
    data['busy_dashers_ratio'] = data['total_busy_dashers'] / data['total_onshift_dashers'].replace(0, np.nan)
    data['total_items_x_busy_dashers_ratio'] = data['total_items'] * data['busy_dashers_ratio']
    data['subtotal_x_busy_dashers_ratio'] = data['subtotal'] * data['busy_dashers_ratio']
    data['log_total_items'] = np.log1p(data['total_items'])
    data['log_subtotal'] = np.log1p(data['subtotal'])
    data['log_avg_item_price'] = np.log1p(data['avg_item_price'])
    data['log_total_delivery_duration'] = np.log1p(data['total_delivery_duration'])
    
    # Handle missing values and outliers
    data.fillna(data.median(), inplace=True)
    max_value_threshold = 1e10
    for column in data.columns:
        data[column] = np.clip(data[column], None, max_value_threshold)
    
    # One-hot encode categorical features
    data = pd.get_dummies(data, columns=['store_primary_category', 'order_protocol', 'market_id'], drop_first=True)
    
    # Drop unnecessary columns
    columns_to_drop = ['created_at', 'actual_delivery_time', 'total_delivery_duration', 'total_items', 'subtotal', 'avg_item_price']
    data.drop(columns=columns_to_drop, inplace=True)
    
    return data

# Train XGBoost model
def train_xgboost(data):
    X = data.drop(columns=['log_total_delivery_duration'])
    y = data['log_total_delivery_duration']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    xgb_model = xgb.XGBRegressor()
    xgb_model.fit(X_train, y_train)
    
    return xgb_model, scaler, X.columns

# Load data and preprocess
data = load_data()
data = preprocess_data(data)

# Train model
xgb_model, scaler, feature_columns = train_xgboost(data)

# Streamlit app
st.title("DoorDash Delivery Time Prediction")

# Data Overview
st.header("Data Overview")
st.write(data.describe())
st.write(data.head())

# Model Training and Evaluation
st.header("Model Training and Evaluation")
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['log_total_delivery_duration']), data['log_total_delivery_duration'], test_size=0.2, random_state=42)
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

st.write(f"Training RMSE: {train_rmse:.4f}, Training R²: {train_r2:.4f}")
st.write(f"Testing RMSE: {test_rmse:.4f}, Testing R²: {test_r2:.4f}")

# Feature Importance
st.header("Feature Importance")
fig, ax = plt.subplots()
xgb.plot_importance(xgb_model, ax=ax)
st.pyplot(fig)

# Prediction Interface
st.header("Predict Delivery Time")
input_data = {
    'total_items': st.number_input('Total Items', min_value=1, max_value=100, value=5),
    'subtotal': st.number_input('Subtotal', min_value=1, max_value=10000, value=5000),
    'total_onshift_dashers': st.number_input('Total Onshift Dashers', min_value=0, max_value=100, value=10),
    'total_busy_dashers': st.number_input('Total Busy Dashers', min_value=0, max_value=100, value=5),
    'total_outstanding_orders': st.number_input('Total Outstanding Orders', min_value=0, max_value=100, value=10),
    'estimated_order_place_duration': st.number_input('Estimated Order Place Duration', min_value=0, max_value=3600, value=300),
    'estimated_store_to_consumer_driving_duration': st.number_input('Estimated Store to Consumer Driving Duration', min_value=0, max_value=3600, value=600),
    'store_primary_category': st.selectbox('Store Primary Category', options=data['store_primary_category'].unique()),
    'order_protocol': st.selectbox('Order Protocol', options=data['order_protocol'].unique()),
    'market_id': st.selectbox('Market ID', options=data['market_id'].unique())
}

input_df = pd.DataFrame([input_data])
input_df['avg_item_price'] = input_df['subtotal'] / input_df['total_items']
input_df['busy_dashers_ratio'] = input_df['total_busy_dashers'] / input_df['total_onshift_dashers'].replace(0, np.nan)
input_df['total_items_x_busy_dashers_ratio'] = input_df['total_items'] * input_df['busy_dashers_ratio']
input_df['subtotal_x_busy_dashers_ratio'] = input_df['subtotal'] * input_df['busy_dashers_ratio']
input_df['log_total_items'] = np.log1p(input_df['total_items'])
input_df['log_subtotal'] = np.log1p(input_df['subtotal'])
input_df['log_avg_item_price'] = np.log1p(input_df['avg_item_price'])

# One-hot encode the categorical features for input data
input_df = pd.get_dummies(input_df, columns=['store_primary_category', 'order_protocol', 'market_id'], drop_first=True)

# Align the input data columns with the training data
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Scale input data
input_scaled = scaler.transform(input_df)

# Predict delivery time
if st.button("Predict"):
    prediction = xgb_model.predict(input_scaled)
    st.write(f"Predicted Log Total Delivery Duration: {prediction[0]:.4f}")
    st.write(f"Predicted Total Delivery Duration (seconds): {np.expm1(prediction[0]):.2f}")
