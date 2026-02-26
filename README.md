ModelBench

Classical Machine Learning vs Deep Learning Benchmark

ModelBench is an interactive Streamlit application that benchmarks classical machine learning models against a simple neural network on the Bank Marketing dataset from the UCI repository.

The goal?
Compare performance, training time, and F1-score between traditional ML and deep learning — on the same dataset, same preprocessing, same split.

📊 Dataset

Bank Marketing Dataset – UCI Repository

The dataset contains customer information from a Portuguese banking institution’s marketing campaign.
The task is binary classification:

Will the customer subscribe to a term deposit? (yes/no)

Target column: y

🧠 Models Compared

The app benchmarks three models:

Logistic Regression

Random Forest

Neural Network (Keras Sequential Model)

Each model is evaluated using:

Accuracy

F1 Score

Training Time

🏗️ Tech Stack

Python

Streamlit

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

Matplotlib

⚙️ How It Works

Dataset is loaded (bank-full.csv)

Categorical features are label-encoded

Data is split into train/test sets (80/20)

Features are standardized

Models are trained

Results are displayed in:

A benchmark comparison table

A visual accuracy bar chart

📂 Project Structure
modelbench/
│
├── app.py
├── bank-full.csv
├── requirements.txt
├── .gitignore
└── README.md

▶️ How To Run Locally
1️⃣ Clone the repository
git clone https://github.com/farhanasabreen-06/modelbench.git
cd modelbench

2️⃣ Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate   # Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run Streamlit app
streamlit run app.py


The app will open in your browser automatically.

📈 What This Project Demonstrates

End-to-end ML pipeline

Model benchmarking

Feature preprocessing

Performance evaluation

Deep learning vs classical ML comparison

Streamlit deployment readiness

🎯 Why This Project Matters

In many real-world tabular datasets:

Classical ML often performs competitively

Deep learning is not always superior

Training time tradeoffs matter

This project shows that clearly and practically.

🚀 Future Improvements

Add more models (XGBoost, SVM)

Hyperparameter tuning

Cross-validation support

Confusion matrix visualization

ROC-AUC comparison

Deployment on Streamlit Cloud / Render

