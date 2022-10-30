import numpy as np


def smape(pred: np.ndarray, true: np.ndarray) -> float:
    denominator = np.abs(true - pred)
    numerator = (np.abs(true) + np.abs(pred)) / 2
    value: float = np.mean(denominator / numerator)
    return value


# TODO: 未完成
# def smape_grad(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
#     # predもtrueも非負であると仮定する
#     denominator = -np.where(true > pred, 1, -1) * 4 * true
#     numerator = (true + pred) ** 2
#     if 0 in numerator:
#         raise ValueError("there are some rows where both of true and value are 0.")
#     value: np.ndarray = denominator / numerator
#     return value


# TODO: 未完成
# def smape_hess(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
#     # predもtrueも非負であると仮定する
#     denominator = +np.where(true > pred, 1, -1) * 8 * true
#     numerator = (true + pred) ** 3
#     if 0 in numerator:
#         raise ValueError("there are some rows where both of true and value are 0.")
#     value: np.ndarray = denominator / numerator
#     return value
