import numpy as np
from loguru import logger

from .data_model import Features, Target
from .model import LgbModel


def train_valid(features: Features, target: Target) -> None:
    model = LgbModel()
    tr_features = features.loc[features.index.str.startswith("tr"), :]
    tr_target = target.loc[target.index.str.startswith("tr")]

    losses: list[float] = []
    for year in tr_features.year.unique():
        trn_idx = tr_features.loc[tr_features.year.ne(year), :].index
        val_idx = tr_features.loc[tr_features.year.eq(year), :].index

        trn_features = tr_features.loc[trn_idx, :]
        trn_target = tr_target.loc[trn_idx]
        val_features = tr_features.loc[val_idx, :]
        val_target = tr_target.loc[val_idx]

        trn_features_pure = trn_features.drop(
            ["date", "year", "month", "year-month"], axis=1
        )
        val_features_pure = val_features.drop(
            ["date", "year", "month", "year-month"], axis=1
        )
        loss = model.train_valid(
            trn_features_pure, trn_target, val_features_pure, val_target
        )
        losses.append(loss)
    logger.info(np.mean(losses))
