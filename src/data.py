import pandas as pd
from sklearn.datasets import load_diabetes

def load_diabetes_df():
    d = load_diabetes(as_frame=True)
    df = d.frame.copy()
    df["target"] = d.target
    return df

# Turn the continuous target into a 0/1 label using the median as the cutoff.

def add_class_label(df):
    median_y = df["target"].median()
    df = df.assign(label=(df["target"] >= median_y).astype(int))
    return df, median_y
