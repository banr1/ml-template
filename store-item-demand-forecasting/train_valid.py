from .data_model import Features, Target
from .model import LgbModel



def train_valid(features: Features, target: Target):
    model = LgbModel()
    tr_features = features.loc[features.index.str.startswith("tr"), :]
    tr_target = target.loc[target.index.str.startswith("tr")]
    model.train_valid(tr_features, tr_target)
