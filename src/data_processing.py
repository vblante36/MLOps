import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path="data/your_dataset.csv", target="target"):
    df = pd.read_csv(path)
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def load_new_data(path="data/new_data.csv"):
    return pd.read_csv(path)