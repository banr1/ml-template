from .data_engineer import data_engineer
from .feature_engineer import feature_engineer
from .train_valid import train_valid


def main() -> None:
    features, target = data_engineer()
    features = feature_engineer(features)
    train_valid(features, target)


if __name__ == "__main__":
    main()
