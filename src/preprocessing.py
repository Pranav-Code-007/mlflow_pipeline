""" Module For Data Preprocessing """

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas as pd

def preprocess_data(df:pd.DataFrame) -> pd.DataFrame:
  """ Preprocess the data """
  return df

def split_data(df:pd.DataFrame, test_size: float, random_state: int = 42) -> list :
    X = df.drop("target", axis=1)
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)






