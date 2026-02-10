🛡️ FraudGuard AI: Real-Time Fraud Detection Console

FraudGuard AI is a machine-learning-powered security console designed to identify fraudulent credit card transactions. Faced with a dataset where fraud represents only 0.17% of activity, this project focuses on high-precision detection and a "Human-in-the-Loop" approach to balance security with customer experience.

🚀 Live Demo

Check out the live dashboard here: https://fraudguard-3znmnb4huqq3kwvrkfyurt.streamlit.app/

🧐 The Challenge: The 0.17% Paradox

In fraud detection, a model that predicts "Safe" for every transaction is 99.83% accurate—yet it catches zero fraud.

This project was built to solve this imbalance by moving away from simple accuracy and focusing on:

Precision (97.2%): Ensuring that when we flag fraud, we are almost certainly correct (minimizing customer friction).

Recall (~75%): Catching as much fraud as possible while maintaining trust.

🛠️ Key Features

Dynamic Threshold Slider: Allows analysts to adjust the model's sensitivity in real-time.

Anomaly Fingerprint: Radar charts that visualize which features (V1-V28) contributed to a suspicious score.

AI Diagnostics: Real-time feedback explaining why the model flagged or missed a specific transaction.

Live Traffic Simulation: Buttons to ingest "Normal" or "Attack" vectors from the local dataset.

📖 Detailed Analysis & Report

For a deep dive into the mathematical framework, model comparison metrics (SMOTE vs. Weighted), and architectural decisions, please refer to:

FraudGuard Project Report.pdf: Full technical documentation and project analysis.

📂 Project Structure

v1_dashboard.py: The main Streamlit dashboard application.

model_fraud.ipynb: Jupyter Notebook containing the initial data exploration, training logic, and model experiments.

fraud_model.joblib: The serialized pre-trained Random Forest model.

scaler.joblibv1: The feature scaler used to normalize transaction data.

Test_Datasetv1.csv: The processed dataset used for the dashboard simulation.

requirements.txt: List of Python dependencies for deployment.

bg_image.jpg: UI background asset for the dashboard.

⚙️ Installation & Usage

Clone the Repo:

git clone [https://github.com/Vishwal-14/fraudguard.git](https://github.com/Vishwal-14/fraudguard.git)
cd fraud_Detection_V1


Install Dependencies:

pip install -r requirements.txt


Run the Dashboard:

streamlit run v1_dashboard.py


🚀 Roadmap (Version 2.0)

. Transition to XGBoost for faster inference and better non-linear pattern recognition.

. Implement Isolation Forest for unsupervised anomaly detection.

. Add Cost-Sensitive Optimization to automatically adjust thresholds based on financial loss projections.

Author: Vishwal

LinkedIn:www.linkedin.com/in/vishwal-sahay-31529a272