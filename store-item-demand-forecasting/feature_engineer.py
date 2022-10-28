from .data_model import Df


def feature_engineer(df: Df) -> Df:
    df["year"] = df.date.dt.year

    return df
