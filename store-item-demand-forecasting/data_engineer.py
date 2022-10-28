import pandas as pd

from .data_model import Df


def data_engineer() -> Df:
    df = pd.read_csv("data/train.csv.zip")
    df["date"] = pd.to_datetime(df.date)
    df = df.astype({"store": "category", "item": "category"})

    return df
