from src.read.read_train_example import read_and_save_intermediate
from src.model import model
from src.datarebuild import data_rebuilder as dr

MODEL = 'model'

REBUILD = 'rebuilt'

READ = 'read'

PERCENTAGE_FILE = .5

ROOT_PATH = "../out/"

DATA_BSON_PATH = '../data/train_example.bson'

DATA_CSV_PATH = "%s/train_example.csv" % ROOT_PATH
DATA_REDUCED_CSV_PATH = "%s/pca_train_example.csv" % ROOT_PATH
DATA_REBUILT_CSV_PATH = "%s/rebuilt/rebuilt-%s.csv" % (ROOT_PATH, PERCENTAGE_FILE)

CATEGORIES_PATH = "%s/csv/categories.csv" % ROOT_PATH
DATA_CSV_PATH_PATTERN = ROOT_PATH + "/csv/full/%s-*.csv"
DATA_CSV_PATH_PATTERN = ROOT_PATH + "/csv/full/%s-*.csv"
DATA_CSV_REBUILT_PATH = ROOT_PATH + "/rebuilt/"

FIRST_OCCURENCE_NUMBER = 1
N_COMPONENTS = 90

pipeline_all = {READ: 1, REBUILD: 1, MODEL: 1}
pipeline_read = {READ: 1, REBUILD: 0, MODEL: 0}
pipeline_rebuild = {READ: 0, REBUILD: 1, MODEL: 0}
pipeline_model = {READ: 0, REBUILD: 0, MODEL: 1}
pipeline_custom = {READ: 1, REBUILD: 0, MODEL: 1}

pipeline = pipeline_all

if __name__ == "__main__":

    if pipeline[READ]:
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

    if pipeline[REBUILD]:
        dr.rebuild_data(
            categories_path=CATEGORIES_PATH,
            percentage=PERCENTAGE_FILE,
            file_pattern=DATA_CSV_PATH_PATTERN,
            output_path=DATA_CSV_REBUILT_PATH
        )

    if pipeline[MODEL]:
        # Build the model from CSV file
        model.do_model(batch_size=15, nb_hidden_layer=10, data_file_path=DATA_REDUCED_CSV_PATH)
