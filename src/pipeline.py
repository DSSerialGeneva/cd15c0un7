from src.read.read_train_example import read_and_save_intermediate
from src.model import model

ROOT_PATH = "../out/"

DATA_BSON_PATH = '../data/train_example.bson'

DATA_CSV_PATH = "%strain_example.csv" % ROOT_PATH
DATA_REDUCED_CSV_PATH = "%spca_train_example.csv" % ROOT_PATH
FIRST_OCCURENCE_NUMBER = 1
N_COMPONENTS = 90

if __name__ == "__main__":
    # Read and create csv files from bson data
    read_and_save_intermediate(
        path=DATA_BSON_PATH,
        pca_reduction=True,
        file_out_path=DATA_CSV_PATH,
        reduced_file_out_path=DATA_REDUCED_CSV_PATH,
        root_path=ROOT_PATH,
        n_components=N_COMPONENTS,
        first_occurence_number=FIRST_OCCURENCE_NUMBER
    )

    # Build the model from CSV file
    model.do_model(batch_size=15, nb_hidden_layer=10, data_file_path=DATA_REDUCED_CSV_PATH)
