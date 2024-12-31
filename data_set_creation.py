import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Generate synthetic data
def create_dataset(num_records=10000):
    np.random.seed(42)

    # Generate features
    customer_ids = np.arange(1, num_records + 1)
    monthly_income = np.random.uniform(2000, 10000, num_records)
    monthly_expenses = np.random.uniform(500, 5000, num_records)
    loan_amount = np.random.uniform(1000, 50000, num_records)
    credit_score = np.random.uniform(300, 850, num_records)
    risk_grade = np.random.choice(['I', 'J', 'K'], size=num_records)
    previous_defaults = np.random.randint(0, 5, num_records)
    default_in_6_months = np.random.choice([0, 1], size=num_records, p=[0.8, 0.2])

    # Create DataFrame
    data = pd.DataFrame({
        'CustomerID': customer_ids,
        'MonthlyIncome': monthly_income,
        'MonthlyExpenses': monthly_expenses,
        'LoanAmount': loan_amount,
        'CreditScore': credit_score,
        'RiskGrade': risk_grade,
        'PreviousDefaults': previous_defaults,
        'DefaultIn6Months': default_in_6_months
    })

    return data

# Save dataset
if __name__ == "__main__":
    dataset = create_dataset()
    dataset.to_csv('credit_default_dataset.csv', index=False)
    print("Dataset created and saved as 'credit_default_dataset.csv'")