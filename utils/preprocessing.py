import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def clean_data(df: pd.DataFrame, features: list) -> pd.DataFrame:
    for feature in features:
        df[feature] = df[feature].astype(float)
        df[feature] = df[feature].fillna(df[feature].mean())
    return df


def encode_text_to_numbers(df: pd.DataFrame, features: list) -> pd.DataFrame:
    le = LabelEncoder()
    for feature in features:
        df[feature] = le.fit_transform(df[feature])
    return df