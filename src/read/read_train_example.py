import numpy
from src.readers import SimpleBSONReader


def read_train_example(path='../../data/train_example.bson'):
    pixels = SimpleBSONReader.read_all(path)
    numpy.savetxt("../../out/train_example.csv", pixels, delimiter=",", fmt='%.d')
    return pixels


def read_and_save_intermediate(path='../../data/train_example.bson'):
    SimpleBSONReader.read_all(path, True)


if __name__ == "__main__":
    # '../../data/train_example.bson'
    read_and_save_intermediate()
