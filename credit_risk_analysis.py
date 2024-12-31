import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_auc_score, matthews_corrcoef

# Load dataset
def load_data(file_path):
    return pd.read_csv(file_path)

# Train and evaluate model
def train_model(data):
    # Define features and target
    X = data.drop(columns=['CustomerID', 'DefaultIn6Months'])
    y = data['DefaultIn6Months']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical features
    categorical_features = ['RiskGrade']

    # Initialize CatBoostClassifier
    model = CatBoostClassifier(iterations=300, learning_rate=0.1, depth=6, auto_class_weights='Balanced', 
                                loss_function='Logloss', cat_features=categorical_features, verbose=0)

    # Train model
    model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=10)

    # Predictions and evaluation
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    pra_auc = roc_auc_score(y_test, y_pred_prob)
    mcc = matthews_corrcoef(y_test, y_pred)

    print(f"PRAUC: {pra_auc:.4f}")
    print(f"MCC: {mcc:.4f}")

    return model

if __name__ == "__main__":
    data = load_data('credit_default_dataset.csv')
    model = train_model(data)
    print("Model training complete.")