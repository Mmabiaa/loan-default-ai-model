import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df):
    df = df.copy()

    # Feature Engineering
    df["debt_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["savings_ratio"] = df["savings_balance"] / (df["loan_amount"] + 1)

    X = df.drop("default", axis=1)
    y = df["default"]

    return X, y