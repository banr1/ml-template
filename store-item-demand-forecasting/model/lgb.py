import lightgbm as lgb

from ..data_model import TrFeatures, TrTarget


class LgbModel:
    def train_valid(self, features: TrFeatures, target: TrTarget) -> None:
        dataset = lgb.Dataset(features, target)
        params = {}
        lgb.train(params, dataset)
