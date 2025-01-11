# Credit Default Prediction Project

## Overview

This project focuses on predicting the likelihood of credit card defaults using machine learning. With financial institutions heavily reliant on accurate predictions to minimize risk and improve profitability, this project provides an automated solution to assess creditworthiness effectively. 

The dataset contains simulated data on customer demographics, financial behavior, and credit utilization. The primary objective is to classify customers into two categories:
1. **Defaulters**: Customers likely to default on their credit obligations.
2. **Non-Defaulters**: Customers likely to meet their credit obligations.

The machine learning model is designed to handle **imbalanced datasets**, where defaulters typically represent a smaller fraction of the data. This is a common scenario in real-world financial data.

---

## Why CatBoost?

The **CatBoost** algorithm was chosen for this project due to the following reasons:

1. **Handling Categorical Data**:
   - Real-world datasets, especially in finance, contain many categorical variables (e.g., marital status, education, employment type). 
   - CatBoost natively supports categorical features, eliminating the need for one-hot encoding or label encoding, thereby reducing preprocessing time and preserving feature interactions.

2. **Performance on Imbalanced Data**:
   - Financial datasets often exhibit class imbalance, with significantly fewer instances of defaulters. CatBoost has inherent capabilities to handle class imbalance effectively.
   - Its **loss functions**, such as `Logloss` and custom metrics like **AUC**, make it ideal for optimizing classification performance.

3. **Fast Training and Inference**:
   - CatBoost is computationally efficient, reducing training time without compromising accuracy.
   - It leverages GPU acceleration and optimized algorithms, making it suitable for large datasets.

4. **High Accuracy and Robustness**:
   - CatBoost consistently achieves high accuracy by preventing overfitting through built-in regularization techniques.
   - Its ability to model complex feature interactions ensures robust predictions.

5. **Ease of Use**:
   - CatBoost is user-friendly and integrates well with Python, making it easy to train and tune the model.

---

## How the Model Helps

The model aids financial institutions by:
- **Risk Assessment**: Automatically identifying high-risk customers who are likely to default.
- **Resource Allocation**: Prioritizing collections and risk mitigation efforts based on predicted defaulters.
- **Revenue Optimization**: Enabling targeted offers to reliable customers and minimizing losses from defaulters.
- **Improving Decision-Making**: Providing actionable insights for underwriting and credit limit adjustments.

---

## Dataset and Features

The dataset used in this project includes the following types of features:
1. **Demographics**: Age, marital status, education level, etc.
2. **Financial Behavior**: Credit utilization, payment history, balance-to-limit ratio.
3. **Account Status**: Number of late payments, credit limit, account age.

These features were chosen based on their relevance to credit risk analysis.

---

## Evaluation Metrics

Since this is a binary classification problem, the following metrics were used to evaluate the model:

1. **Accuracy**: Measures the overall correctness of predictions but may not be sufficient for imbalanced datasets.
2. **Precision**: Focuses on the proportion of correctly identified defaulters among all predicted defaulters.
3. **Recall (Sensitivity)**: Ensures that the model identifies most of the actual defaulters, critical for reducing false negatives.
4. **F1-Score**: Balances precision and recall, offering a comprehensive measure of performance.
5. **ROC-AUC**: Provides a summary of model performance across various classification thresholds, useful for imbalanced data.
6. **Matthews Correlation Coefficient (MCC)**: Offers a balanced evaluation metric, particularly useful for imbalanced datasets.

---

## Workflow of the Project

### 1. **Data Preparation**
- The dataset is either sourced or generated using synthetic data creation scripts.
- Missing values and outliers are handled to ensure data quality.
- Features are categorized into numerical and categorical for appropriate processing.

### 2. **Feature Engineering**
- Derived features such as balance-to-limit ratio are added to improve model accuracy.
- Categorical features are automatically processed by CatBoost.

### 3. **Model Training**
- The CatBoost model is trained on a labeled dataset with default and non-default instances.
- Hyperparameters, such as learning rate, tree depth, and iterations, are optimized using grid search or Bayesian optimization.

### 4. **Evaluation and Validation**
- The model is evaluated on a test dataset to ensure it generalizes well to unseen data.
- Metrics like AUC-ROC and F1-Score are closely monitored to assess its effectiveness.

### 5. **Deployment and Usage**
- The trained model is saved as a `pkl` file for future inference.
- During deployment, the model takes customer data as input and outputs the probability of default.

---

## Applications of the Model

1. **Credit Underwriting**:
   - Predict creditworthiness during loan or credit card application processes.

2. **Risk Monitoring**:
   - Continuously monitor existing customers to detect potential risks and take proactive measures.

3. **Fraud Detection**:
   - Identify unusual patterns that might signal fraudulent activities or risky behavior.

4. **Customer Segmentation**:
   - Segment customers based on risk profiles to offer personalized services.

---

## Dependencies

- **Python**: The entire project is implemented in Python for its versatility and the availability of data science libraries.
- **Libraries Used**:
  - **CatBoost**: For training and evaluating the machine learning model.
  - **Pandas**: For data manipulation and analysis.
  - **NumPy**: For numerical computations.
  - **Scikit-learn**: For evaluation metrics and model comparison.
  - **Matplotlib**: For visualizing model performance and dataset insights.

---

## Future Scope

1. **Integration with Real-Time Systems**:
   - Deploy the model on cloud platforms to enable real-time credit risk analysis.

2. **Explainability**:
   - Use SHAP (SHapley Additive exPlanations) to explain predictions to stakeholders.

3. **Model Improvements**:
   - Experiment with other models like XGBoost or LightGBM for comparison.
   - Add additional features such as transaction history and external credit bureau scores.

4. **Scalability**:
   - Handle larger datasets using distributed training on platforms like Spark.

---

## How to Run the Project

### 1. Clone the Repository
```bash
git clone https://github.com/mtptisid/Credit-Risk-Analysis-with-CatBoostClassifier
cd Credit-Risk-Analysis-with-CatBoostClassifier
```

### 2. Set Up the Environment

Ensure Python and required libraries are installed. You can use a virtual environment:

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\\Scripts\\activate
pip install -r requirements.txt
```

### 3. Create the Dataset

Run the dataset generation script to create `credit_default_dataset.csv`:

```bash
python create_dataset.py
```

### 5. Evaluate the Model

View the evaluation metrics to assess model performance.

## Dependencies

- Python >= 3.8
- Libraries: catboost, pandas, scikit-learn, numpy, matplotlib

## Scripts Overview

1. **create_dataset.py**: Generates synthetic data for credit default prediction.
2. **train_model.py**: Trains a CatBoost classifier on the generated dataset.

## Screenshots:

<img width="1440" alt="Screenshot 2025-01-01 at 3 52 15 AM" src="https://github.com/user-attachments/assets/3061850e-159a-49a4-8f18-16daa41690b3" />

## Contact

For any queries, feedback, or collaboration opportunities, please reach out:

- **Name**: Siddharamayya M.
- **Email**: [msidrm455@gmail.com](mailto:msidrm455@gmail.com)
- **Portfolio**: [My Portfolio](https://mtptisid.github.io)
