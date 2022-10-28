# ML Template
個人的な機械学習プロジェクトのテンプレート。

例として「Store Item Demand Forecasting Challenge」のデータを用いた需要予測のプロジェクトと仮定する。
https://www.kaggle.com/competitions/demand-forecasting-kernels-only

## 命名規則
`pd.DataFrame` の命名規則
- インデックス: 学習データかテストデータか、と連番(6桁)からなる
    - ex) `tr000001`
