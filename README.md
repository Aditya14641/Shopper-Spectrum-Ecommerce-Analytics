# 🛒 Shopper Spectrum: E-Commerce Customer Segmentation & Recommendations

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Unsupervised-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Clustering-yellow)

## 📌 Project Overview
The global e-commerce industry generates vast amounts of transaction data daily. This capstone project analyzes historical transaction logs from an online retail business to uncover hidden purchasing patterns, logically group the customer base, and cross-sell items. 

By engineering a **Recency, Frequency, and Monetary (RFM)** framework, this project utilizes **K-Means Clustering** to segment customers into actionable business tiers. Additionally, an **Item-Based Collaborative Filtering** recommendation engine was built using Cosine Similarity to suggest "frequently bought together" products, maximizing average order value. The entire machine learning pipeline is deployed as an interactive **Streamlit Web Application**.

## 🎯 Problem Statement
E-commerce platforms often struggle to personalize marketing efforts due to a lack of structured customer understanding. The objective of this project is to:
1. Clean and analyze raw transactional data to find macro business trends.
2. Automatically segment the customer base using unsupervised machine learning.
3. Develop a recommendation system that suggests similar products to users.
4. Deploy these models into a user-friendly UI for non-technical business stakeholders.

## 🚀 Key Features & Modules
* **Exploratory Data Analysis (EDA):** Comprehensive visualizations of top-selling products, revenue by country, peak order hours, and behavioral distributions using Seaborn and Matplotlib.
* **Customer Segmentation Module:** Takes live user inputs for Recency, Frequency, and Monetary values, scales the data, and predicts the customer's loyalty segment (e.g., *High-Value VIP, Regular, Occasional, At-Risk*) using a pre-trained K-Means model.
* **Product Recommendation Module:** Allows users to search for any product in the catalog and instantly outputs the top 5 most similar products based on historical co-purchasing data (Cosine Similarity).

## 🛠️ Technology Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means, StandardScaler, Cosine Similarity)
* **Data Visualization:** Matplotlib, Seaborn
* **Deployment & UI:** Streamlit
* **Serialization:** Joblib, Pickle

## 📂 Repository Structure
```text
📦 Shopper-Spectrum-Ecommerce-Analytics
 ┣ 📜 project_eda.ipynb  # Comprehensive Exploratory Data Analysis
 ┣ 📜 project_ml.ipynb   # Machine Learning modeling and serialization
 ┣ 📜 app.py                                # Streamlit Web Application script
 ┣ 📜 requirements.txt                      # Python dependencies
 ┣ 📜 kmeans_model.pkl                      # Serialized K-Means clustering model
 ┣ 📜 scaler.pkl                            # Serialized Standard Scaler
 ┣ 📜 item_similarity.pkl                   # Serialized Cosine Similarity dataframe
 ┗ 📜 README.md                             # Project documentation
```
## 💻 Installation & Setup
To run this project locally on your machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/Shopper-Spectrum-Ecommerce-Analytics.git](https://github.com/your-username/Shopper-Spectrum-Ecommerce-Analytics.git)
cd Shopper-Spectrum-Ecommerce-Analytics
```
**2. Create a virtual environment (Recommended):**
```bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Mac/Linux:
source venv/bin/activate
```
**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```
**4. Run the Streamlit Application:**
```bash
streamlit run app.py
```
## 💡 How to Use the App
1. **Product Recommendations:** Navigate to the top module of the app. Use the dropdown menu to search for or select a product name. Click "Get Recommendations" to see the top 5 items most frequently bought alongside it.
2. **Customer Segmentation:** Navigate to the bottom module. Enter the number of days since a customer's last purchase (Recency), their total number of orders (Frequency), and their total lifetime spend (Monetary). Click "Predict Cluster" to see which marketing segment they belong to.
