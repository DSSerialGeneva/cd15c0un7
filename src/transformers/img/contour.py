import numpy as np
import matplotlib.pyplot as plt


import io

import bson

from PIL import Image

from matplotlib.mlab import PCA

from src.domain.img_meta import ImgMeta
from src.transformers.img.img_2_array_gray import Img2ArrayGray

from skimage import measure


def contour_sample():
    x, y = np.ogrid[-np.pi: np.pi: 100j, -np.pi:np.pi: 100j]
    r = np.sin(np.exp((np.sin(x) ** 3 + np.cos(y) ** 2)))
    return r


def from_cdiscount_image(file_name='../../../data/train_example.bson'):
    bson_file = open(file_name, 'rb')

    bson_iterator = bson.decode_file_iter(bson_file)

    for c, d in enumerate(bson_iterator):

        for no_pic, pic in enumerate(d['imgs']):
            img = Image.open(io.BytesIO(pic['picture']))
            break
        break
    return np.array(img.convert('L'))


def contour_from_array(np_array):

    contours = measure.find_contours(np_array, 0.8)

    fig, ax = plt.subplots()

    ax.imshow(np_array, interpolation='nearest', cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


def show_img(np_array):
    fig, ax = plt.subplots()

    ax.imshow(np_array, interpolation='nearest', cmap=plt.cm.gray)

    ax.axis('image')

    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()


#from_cdiscount_image()

#contour_from_array(from_cdiscount_image())

#show_img(from_cdiscount_image())

pca = PCA(from_cdiscount_image())
show_img(pca)
