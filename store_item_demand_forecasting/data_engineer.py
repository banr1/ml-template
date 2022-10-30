import pandas as pd

from .data_model import Features, Target


def set_index(df: pd.DataFrame, prefix: str) -> Features:
    df = df.reset_index()
    df["index"] = prefix + df.index.astype("str").str.zfill(6)
    df = df.rename({"index": "idx"}, axis=1).set_index("idx")
    return df


def data_engineer() -> tuple[Features, Target]:
    train = pd.read_csv("data/train.csv.zip")
    test = pd.read_csv("data/test.csv").drop("id", axis=1)
    train = set_index(train, prefix="tr")
    test = set_index(test, prefix="ts")
    data = pd.concat([train, test])
    data["date"] = pd.to_datetime(data.date)
    data = data.astype({"store": "category", "item": "category"})
    target = data.sales
    features = data.drop("sales", axis=1)

    features.to_csv("data/cache/data_engineered/features.csv.zip")
    target.to_csv("data/cache/data_engineered/target.csv.zip")

    return features, target
