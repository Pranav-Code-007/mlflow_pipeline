""" 
Module for data loading 
"""

import pandas as pd
from sklearn.datasets import load_iris

def load_data() -> pd.DataFrame:
    """ Load the Iris dataset and return as a DataFrame """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df
