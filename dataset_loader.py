import pandas as pd
def load_data():
    df = pd.read_csv("data.csv")
    X = df.iloc[:, 2:].values
    y = df.iloc[:, 1].values
    return X, y