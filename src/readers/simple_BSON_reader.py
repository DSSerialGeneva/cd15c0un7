import csv
import hashlib
import io
import operator
import os

import bson
import numpy as np
import pandas as pd
from PIL import Image

from src.domain.img_meta import ImgMeta
from src.transformers.img.img_2_array_gray import Img2ArrayGray

FIRST_OCCURENCE_NUMBER = 1
ROOT_DIR = "../../out/csv/"


class ParsingResult:
    pixel_matrix = np.empty(())
    categories = dict()  # to list and count the categories
    products = dict()  # to list and count the products
    pictures_categories = np.empty(())  # to list and count the couples (picture hash, category)
    pictures = dict()  # to list and count the pictures hashes
    twin_pictures = dict()  # to list and count the pictures which are twice with a different categories


class SimpleBSONReader:

    @staticmethod
    def read_first(file_name):
        bson_file = open(file_name, 'rb')

        image_bytes = bson.loads(bson_file.read())['imgs'][0]['picture']

        stream = io.BytesIO(image_bytes)

        return Image.open(stream)

    @staticmethod
    def increment_counter_dict(counters, key):
        counters[key] = counters.get(key, 0) + 1

    @staticmethod
    def increment_counter_array(counters, key):
        counters[key] = counters.get(key, 0) + 1

    @staticmethod
    def read_all(file_name, save_intermediate=False):

        if save_intermediate:
            if not os.path.exists(ROOT_DIR):
                os.mkdir(ROOT_DIR)

        bson_file = open(file_name, 'rb')

        bson_iterator = bson.decode_file_iter(bson_file)

        parsing_result = ParsingResult()
        for c, d in enumerate(bson_iterator):
            product_id = d['_id']
            category_id = d['category_id']
            for no_pic, pic in enumerate(d['imgs']):
                SimpleBSONReader.increment_counter_dict(parsing_result.categories, category_id)
                SimpleBSONReader.increment_counter_dict(parsing_result.products, product_id)

                picture = Image.open(io.BytesIO(pic['picture']))

                SimpleBSONReader.picture_hash_count(
                    category_id,
                    pic,
                    parsing_result)

                img_meta = ImgMeta(product_id, category_id, no_pic)

                flatten_ = img_meta.__array__() + tuple(Img2ArrayGray.transform(picture, True, img_meta).flatten())
                if parsing_result.pixel_matrix.shape == ():
                    parsing_result.pixel_matrix = np.array(flatten_)
                else:
                    parsing_result.pixel_matrix = np.vstack((
                        parsing_result.pixel_matrix,
                        flatten_
                    ))
            if save_intermediate:
                f = open(ROOT_DIR + "%s-%s.csv" % (category_id, product_id), 'w')
                f.write(",".join(map(str, flatten_)))
        if save_intermediate:
            with open(ROOT_DIR + "total.csv", 'w') as f:
                f.write("%i" % c)
            SimpleBSONReader.write_dict_to_file("categories.csv", parsing_result.categories)
            SimpleBSONReader.write_dict_to_file("products.csv", parsing_result.products)
            SimpleBSONReader.write_dict_to_file("pictures.csv", parsing_result.pictures)

            df = pd.DataFrame(parsing_result.pictures_categories)
            SimpleBSONReader.write_df_to_file("pictures_categories.csv", df)
            df = df[df[2].map(int) != FIRST_OCCURENCE_NUMBER]
            SimpleBSONReader.write_df_to_file("pictures_categories_cleaned.csv", df)
        return parsing_result.pixel_matrix

    @staticmethod
    def write_dict_to_file(file_name, array):
        with open(ROOT_DIR + file_name, 'w') as file:
            SimpleBSONReader.write_csv_dictionary(file, array)

    @staticmethod
    def write_df_to_file(file_name, array):
        with open(ROOT_DIR + file_name, 'w') as file:
            SimpleBSONReader.write_csv_df(file, array)

    @staticmethod
    def picture_hash_count(category_id, pic, parsing_result):
        md5 = hashlib.md5()
        md5.update(pic['picture'])
        md5_strip = md5.hexdigest().strip()
        category_id = str(category_id)
        pict_cat_row = [category_id, md5_strip, FIRST_OCCURENCE_NUMBER]
        if parsing_result.pictures_categories.shape == ():
            is_new_category = False  # because we don't want to mark the category as new when the picture is new too
            if parsing_result.pictures_categories.shape == ():
                parsing_result.pictures_categories = np.array(pict_cat_row)
            else:
                parsing_result.pictures_categories = np.vstack((parsing_result.pictures_categories, pict_cat_row))
        else:
            if parsing_result.pictures_categories.shape == (3,):
                # first row
                same_pictures = np.nonzero(parsing_result.pictures_categories[1] == md5_strip)[0]
            else:
                # other rows
                same_pictures = np.nonzero(parsing_result.pictures_categories[:, 1] == md5_strip)[0]
            not_new_picture = len(same_pictures) > 0
            is_new_category = True
            if is_new_category and not_new_picture:
                # we'll find if it's a new category for the existing same pictures
                for i_same_picture in same_pictures:
                    same_category = parsing_result.pictures_categories[i_same_picture, 0] == category_id
                    if same_category:
                        # when category is the same, increase the counter, set new category to false
                        is_new_category = False
                        parsing_result.pictures_categories[i_same_picture, 2] = 1 + int(parsing_result.pictures_categories[i_same_picture, 2])
                        break
            else:
                is_new_category = False  # because we don't want to mark the category as new when the picture is new too
                if parsing_result.pictures_categories.shape == ():
                    parsing_result.pictures_categories = np.array(pict_cat_row)
                else:
                    parsing_result.pictures_categories = np.vstack((parsing_result.pictures_categories, pict_cat_row))
        if is_new_category:
            SimpleBSONReader.increment_counter_dict(parsing_result.twin_pictures, md5_strip)
        SimpleBSONReader.increment_counter_dict(parsing_result.pictures, md5_strip)

    @staticmethod
    def write_csv_dictionary(csv_file, dictionary):
        csv.writer(csv_file).writerows(sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True))

    @staticmethod
    def write_csv_df(csv_file, df):
        df.to_csv(csv_file, index=False, header=False)
