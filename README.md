# Credit Default Prediction Using Machine Learning

## Project Overview

This project is designed to predict the likelihood of credit default among professional customers of a financial institution.  
By identifying potential defaulters early, financial institutions can mitigate risks, improve profitability, and enhance customer service.

## Table of Contents

- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Dataset Details](#dataset-details)
- [Model Used](#model-used)
- [Performance Metrics](#performance-metrics)
- [How to Run the Project](#how-to-run-the-project)
- [Future Improvements](#future-improvements)

## Introduction

The objective is to build a machine learning model to classify whether a customer is likely to default on credit payments within the next six months. The solution includes:
- Dataset creation and preprocessing.
- A CatBoost classification model for predictions.
- Evaluation using robust metrics suitable for imbalanced datasets.

## Technologies Used

- Python 3.8+
- Libraries: 
  - pandas
  - numpy
  - sklearn
  - catboost

## Dataset Details

The dataset is synthetically generated and consists of the following features:
- **CustomerID**: Unique identifier for each customer.
- **MonthlyIncome**: The customer's monthly income.
- **MonthlyExpenses**: The customer's monthly expenses.
- **LoanAmount**: Total loan amount taken by the customer.
- **CreditScore**: Customer's credit score (300 to 850 scale).
- **RiskGrade**: Categorical grade indicating risk (I, J, K).
- **PreviousDefaults**: Number of previous defaults by the customer.
- **DefaultIn6Months**: Target variable (0 = No default, 1 = Default).

## Model Used

The **CatBoost Classifier** was chosen for its:
- Superior handling of categorical data.
- Robust performance on tabular datasets.
- Built-in regularization to prevent overfitting.

## Performance Metrics

- **PRAUC (Precision-Recall Area Under Curve)**: Focuses on identifying defaults in imbalanced datasets.
- **MCC (Matthews Correlation Coefficient)**: Evaluates balanced performance across all confusion matrix categories.

## How to Run the Project

### 1. Clone the Repository
```bash
git clone <repository_link>
cd <repository_name>
```