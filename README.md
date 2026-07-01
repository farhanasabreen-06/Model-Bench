# ModelBench

### Benchmarking Classical Machine Learning vs Deep Learning Models

ModelBench is an interactive benchmarking application built to compare the performance of **Classical Machine Learning** and **Deep Learning** models using structured evaluation metrics.

The project enables users to train, evaluate, and analyze multiple models on the **UCI Bank Marketing Dataset** and compare their effectiveness through visual insights and performance analysis.

---

##  Features

- Compare Classical ML and Deep Learning approaches
- Interactive web interface using Streamlit
- Automated model training and evaluation
- Performance comparison using:
      * Accuracy
      * F1 Score
      * Training Time

- Visual representation of results
- Clean and easy-to-use workflow

---

## 77 Tech Stack

* **Python**
* **Streamlit**
* **Scikit-learn**
* **TensorFlow**
* **Pandas**
* **NumPy**
* **Matplotlib**

---

##  Project Structure

```plaintext
modelbench/
│
├── app.py                  # Main Streamlit application
├── requirements.txt        # Project dependencies
├── dataset/                # Dataset files
├── models/                 # Model implementations
├── utils/                  # Helper functions
└── README.md               # Project documentation
```

---

##  Installation

Clone the repository:

```bash
git clone <repository-url>
```

Move into the project folder:

```bash
cd modelbench
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

##  Evaluation Metrics

The models are benchmarked using:

| Metric        | Purpose                            |
| ------------- | ---------------------------------- |
| Accuracy      | Measures prediction correctness    |
| F1 Score      | Balances precision and recall      |
| Training Time | Evaluates computational efficiency |

---

##  Objective

The objective of ModelBench is to provide a practical environment for understanding the trade-offs between traditional Machine Learning techniques and Deep Learning approaches through measurable performance evaluation.

---

##  Future Improvements

* Add more ML and DL algorithms
* Hyperparameter tuning support
* Dataset upload feature
* Model export functionality
* Advanced visual analytics dashboard

---


⭐ If you found this project useful, consider giving it a star.

