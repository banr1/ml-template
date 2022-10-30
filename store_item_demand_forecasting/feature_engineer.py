from .data_model import Features


def feature_engineer(features: Features) -> Features:
    features["year"] = features.date.dt.year
    features["month"] = features.date.dt.month
    features["dow"] = features.date.dt.dayofweek.astype("category")
    features["year-month"] = (
        features.year.astype("str") + "-" + features.month.astype("str").str.zfill(2)
    )
    features["store-item"] = (
        features.store.astype("str").str.zfill(2)
        + "-"
        + features.item.astype("str").str.zfill(2)
    ).astype("category")

    features.to_csv("data/cache/feature_engineered/features.csv.zip")

    return features
