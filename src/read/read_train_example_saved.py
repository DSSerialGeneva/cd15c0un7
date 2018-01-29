import pandas as pd


def read_train_example_saved(path="../../out/train_example.csv"):
    return pd.read_csv(path, quotechar='"').iloc[:, :].values


if __name__ == "__main__":
    read_train_example_saved()
