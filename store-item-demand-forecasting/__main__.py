from .data_engineer import data_engineer
from .feature_engineer import feature_engineer


def main():
    data = data_engineer()
    feature = feature_engineer(data)


if __name__ == "__main__":
    main()
