import pandas as pd
import numpy as np
import os

def generate_datasets():
    data_dir = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(data_dir, exist_ok=True)
    n_samples = 100

    # 1. Real Estate (Prediction vs Inference)
    np.random.seed(42)
    distance_to_city = np.random.uniform(1, 50, n_samples)
    house_size = np.random.uniform(500, 5000, n_samples)
    price = 500000 - (distance_to_city * 5000) + (house_size * 200) + np.random.normal(0, 20000, n_samples)
    pd.DataFrame({"Distance_to_City_Miles": distance_to_city, "House_Size_sqft": house_size, "Price_USD": price}).to_csv(os.path.join(data_dir, "Real_Estate_Dataset.csv"), index=False)

    # 2. Customer LTV (Prediction vs Inference)
    np.random.seed(43)
    months_active = np.random.uniform(1, 60, n_samples)
    support_tickets = np.random.poisson(lam=2, size=n_samples)
    ltv = (months_active * 150) - (support_tickets * 50) + np.random.normal(0, 100, n_samples)
    pd.DataFrame({"Months_Active": months_active, "Support_Tickets": support_tickets, "LTV_USD": ltv}).to_csv(os.path.join(data_dir, "Customer_LTV_Dataset.csv"), index=False)

    # 3. Factory Output (Parametric vs Non-Parametric) - Non-linear
    np.random.seed(44)
    machine_temp = np.random.uniform(50, 150, n_samples)
    # Output crashes if temp is too high or too low (U-shape upside down)
    output = 100 - ((machine_temp - 100)**2) * 0.05 + np.random.normal(0, 5, n_samples)
    pd.DataFrame({"Machine_Temp_C": machine_temp, "Output_Units": output, "True_f": 100 - ((machine_temp - 100)**2) * 0.05}).to_csv(os.path.join(data_dir, "Factory_Output_Dataset.csv"), index=False)

    # 4. Employee Productivity (Parametric vs Non-Parametric) - Highly non-linear step function
    np.random.seed(45)
    training_hours = np.random.uniform(0, 100, n_samples)
    prod = np.piecewise(training_hours, [training_hours < 20, (training_hours >= 20) & (training_hours < 60), training_hours >= 60], [30, 70, 95]) + np.random.normal(0, 5, n_samples)
    pd.DataFrame({"Training_Hours": training_hours, "Productivity_Score": prod}).to_csv(os.path.join(data_dir, "Employee_Productivity_Dataset.csv"), index=False)

    # 5. Loan Approval (Classification)
    np.random.seed(46)
    credit_score = np.random.uniform(300, 850, n_samples)
    income = np.random.uniform(20000, 150000, n_samples)
    score = (credit_score - 600)/100 + (income - 50000)/30000
    prob = 1 / (1 + np.exp(-score))
    approved = np.random.binomial(1, prob)
    pd.DataFrame({"Credit_Score": credit_score, "Income_USD": income, "Approved": approved}).to_csv(os.path.join(data_dir, "Loan_Approval_Dataset.csv"), index=False)

    # 6. Customer Churn (Classification) - Circular boundary
    np.random.seed(47)
    usage_frequency = np.random.uniform(0, 100, n_samples)
    price_sensitivity = np.random.uniform(0, 100, n_samples)
    dist = (usage_frequency - 50)**2 + (price_sensitivity - 50)**2
    churn_prob = 1 / (1 + np.exp((dist - 900)/100)) # Churn if in the middle
    churn = np.random.binomial(1, churn_prob)
    pd.DataFrame({"Usage_Frequency": usage_frequency, "Price_Sensitivity": price_sensitivity, "Churn": churn}).to_csv(os.path.join(data_dir, "Customer_Churn_Dataset.csv"), index=False)

if __name__ == "__main__":
    generate_datasets()
    print("6 New Business Datasets generated successfully.")
