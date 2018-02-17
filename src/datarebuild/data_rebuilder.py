import pandas as pd
import glob


def rebuild_data(
        categories_path,
        percentage=0.50,
        file_pattern='../../out/csv/full/%s-*.csv', output_path='../../out/rebuilt/'):

    output_path = output_path + 'rebuilt-%s.csv' % percentage

    # extract categories keeping only the first {percentage} part
    sub_categories = extract_categories(categories_path, percentage)

    # for every chosen category, add the row in the csv file
    with open(output_path, mode='w') as f:
        for sub_category in sub_categories:
            chosen_files_paths = glob.glob(file_pattern % sub_category)
            for chosen_file_path in chosen_files_paths:
                with open(chosen_file_path, 'r') as chosen_file:
                    f.write(chosen_file.readline())
                    f.flush()


def extract_categories(categories_path, percentage):
    categories = pd.read_csv(categories_path, index_col=False, names=('category_id', 'count'))
    sub_categories = categories.loc[0:(categories.shape[0] * percentage), 'category_id']
    return sub_categories


if __name__ == "__main__":
    rebuild_data('../../out/csv/categories.csv', percentage=.30)
