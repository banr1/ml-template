import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data_model import TrFeatures, TrTarget
from ..loss import smape


class LgbModel:
    def train_valid(
        self,
        trn_features: TrFeatures,
        trn_target: TrTarget,
        val_features: TrFeatures,
        val_target: TrTarget,
    ) -> float:
        trn_target_mean = int(trn_target.mean())
        trn_init_score = pd.Series(
            [trn_target_mean for _ in range(len(trn_target))], index=trn_target.index
        )
        val_init_score = pd.Series(
            [trn_target_mean for _ in range(len(val_target))], index=val_target.index
        )
        trn_dataset = lgb.Dataset(trn_features, trn_target, init_score=trn_init_score)
        val_dataset = lgb.Dataset(val_features, val_target, init_score=val_init_score)
        params = {
            "learning_rate": 0.1,
            "objective": "regression",
            "metric": "None",
            "seed": 0,
        }
        model = lgb.train(
            params,
            trn_dataset,
            valid_sets=val_dataset,
            valid_names="valid",
            feval=self._smape,
        )
        best_smape: float = model.best_score["valid"]["SMAPE"]
        return best_smape

    def _smape(self, preds: np.ndarray, data: lgb.Dataset) -> tuple[str, float, bool]:
        trues = data.get_label()
        value = smape(preds, trues)

        return "SMAPE", value, False

    # TODO: 未完成
    # def _smape_for_objective(self, preds: np.ndarray, data: lgb.Dataset):
    #     trues = data.get_label()

    #     grad = smape_grad(preds, trues)
    #     hess = smape_hess(preds, trues)

    #     return grad, hess
