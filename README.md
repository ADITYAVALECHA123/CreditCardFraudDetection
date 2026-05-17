# Credit Card Fraud Detection System
## Overview

This project is an end-to-end Machine Learning based Credit Card Fraud Detection System developed using Python, Machine Learning, Deep Learning, and Streamlit.

The system analyzes transaction-related information and predicts whether a transaction is fraudulent or legitimate using trained ML/DL models.

### The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Machine Learning Models
- Deep Learning Model
- Ensemble Learning
- Streamlit Web Application
- Cloud Deployment

### Features
- Real-time fraud prediction
- Machine Learning pipeline
- Ensemble model support
- ANN (Artificial Neural Network)
- Fraud probability detection
- Interactive Streamlit UI
- SHAP explainability support
- Cloud deployment using Streamlit Cloud

### Technologies Used
#### Programming Language
- Python
#### Libraries
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- LightGBM
- SHAP
- Matplotlib
- Seaborn
- Plotly
- Streamlit
- Joblib
- Optuna

### Machine Learning Workflow
1. Data Collection
The dataset contains transaction-related information such as:
- Merchant
- Category
- City
- State
- Job
- Transaction Amount
- Gender
- Latitude/Longitude
- Date & Time

2. Data Preprocessing
- Missing value handling
- Feature encoding
- Feature scaling
- Datetime feature extraction
- Outlier handling

3. Exploratory Data Analysis (EDA)<br>
Performed detailed analysis including:
- Fraud distribution
- Transaction patterns
- Correlation analysis
- State-wise fraud analysis
- Merchant/category analysis
- Time-based fraud trends

4. Feature Engineering
Created advanced features from:
- Transaction time
- Location data
- User behavior
- Merchant patterns

5. Models Used
#### Machine Learning Models
- Random Forest
- XGBoost
- LightGBM
#### Deep Learning
- Artificial Neural Network (ANN)

#### Ensemble Learning<br>
Combined multiple models for better fraud detection performance.

#### Model Evaluation Metrics<br>
The models were evaluated using:
Accuracy
Precision
Recall
F1 Score
ROC-AUC
PR-AUC

Streamlit Web Application
The project includes an interactive Streamlit frontend where users can:

Enter transaction details
Predict fraud probability
View fraud alerts
Analyze predictions

### Installation
1. Clone Repository<br>
git clone https://github.com/YOUR_USERNAME/CreditCardFraudDetection.git

2. Move to Project Folder<br>
cd CreditCardFraudDetection

3. Create Virtual Environment
Windows<br>
python -m venv venv

Activate:<br>
venv\Scripts\activate

4. Install Requirements
pip install -r requirements.txt

5. Run Streamlit App
streamlit run frontend/app.py
