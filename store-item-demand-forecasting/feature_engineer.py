from .data_model import Features


def feature_engineer(df: Features) -> Features:
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month

    return df
