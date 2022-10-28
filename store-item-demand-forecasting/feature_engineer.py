from .data_model import Features


def feature_engineer(df: Features) -> Features:
    df["year"] = df.date.dt.year
    df = df.drop("date", axis=1)

    return df
