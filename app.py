import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(page_title="Shopper Spectrum App", layout="wide")
st.title("🛒 Shopper Spectrum: E-Commerce Analytics")

# Load Serialized Models safely
@st.cache_resource
def load_models():
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    item_sim = pd.read_pickle('item_similarity.pkl')
    return kmeans, scaler, item_sim

try:
    kmeans_model, scaler_model, item_similarity_df = load_models()
except Exception as e:
    st.error("Model files not found. Please ensure the .pkl files are in the same directory as app.py")
    st.stop()

# ----- MODULE 1: PRODUCT RECOMMENDATION -----
st.header("🎯 1️⃣ Product Recommendation Module")
st.write("Using Item-Based Collaborative Filtering")

# Searchable dropdown for products
product_list = item_similarity_df.columns.tolist()
selected_product = st.selectbox("Select or Search for a Product Name:", product_list)

if st.button("Get Recommendations"):
    # Get top 5 similar products (ignoring the 1st one which is the product itself)
    recommendations = item_similarity_df[selected_product].sort_values(ascending=False)[1:6]
    
    st.success(f"Customers who bought **{selected_product}** also bought:")
    for idx, item in enumerate(recommendations.index):
        st.write(f"{idx+1}. {item} (Similarity Score: {recommendations[item]:.2f})")


st.divider()

# ----- MODULE 2: CUSTOMER SEGMENTATION -----
st.header("🎯 2️⃣ Customer Segmentation Module")
st.write("Predict a customer's segment using live RFM inputs.")

col1, col2, col3 = st.columns(3)
with col1:
    r_input = st.number_input("Recency (Days since last purchase):", min_value=0, value=30)
with col2:
    f_input = st.number_input("Frequency (Total distinct orders):", min_value=1, value=5)
with col3:
    m_input = st.number_input("Monetary (Total $ spent):", min_value=0.0, value=250.0)

if st.button("Predict Cluster"):
    # Scale the inputs
    input_data = pd.DataFrame([[r_input, f_input, m_input]], columns=['Recency', 'Frequency', 'Monetary'])
    scaled_data = scaler_model.transform(input_data)
    
    # Predict the cluster
    cluster_id = kmeans_model.predict(scaled_data)[0]
    
    # Define business logic mappings based on standard RFM structures
    segment_labels = {
        0: "At-Risk / Churning Customer",
        1: "Regular Customer",
        2: "High-Value / VIP Customer",
        3: "Occasional / New Customer"
    }
    
    # Fallback if your specific model generated different clusters
    label = segment_labels.get(cluster_id, f"Cluster {cluster_id}")
    
    st.info(f"🔮 The model predicts this user belongs to: **{label}**")