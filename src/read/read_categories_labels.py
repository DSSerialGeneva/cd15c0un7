import pandas as pd


def read_categories_labels(path="../../data/category_names.csv"):
    return pd.read_csv(path, quotechar='"').iloc[:, 0].values


if __name__ == "__main__":
    categories = read_categories_labels()
    print(type(categories))
    print(categories[:])

