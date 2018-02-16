import csv
import hashlib
import io
import operator
import bson
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import pca
from PIL import Image

from src.domain.img_meta import ImgMeta
from src.transformers.img.img_2_array_gray import Img2ArrayGray


class ParsingResult:
    pixel_matrix = np.empty(())
    categories = dict()  # to list and count the categories occurrences (nb of pictures)
    products = dict()  # to list and count the products

    # to list and count the couples (picture hash, category)
    pictures_categories = pd.DataFrame(columns=['category', 'pic_hash', 'count'])
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
    def read_all(
            file_name,
            save_intermediate=False,
            save_png=True,
            pca_reduction=True,
            root_path="../../out",
            n_components=90, first_occurence_number=1):

        bson_file = open(file_name, 'rb')

        bson_iterator = bson.decode_file_iter(bson_file)

        parsing_result = ParsingResult()
        for total_pic_rows, d in enumerate(bson_iterator):
            product_id = d['_id']
            category_id = d['category_id']
            for no_pic, pic in enumerate(d['imgs']):
                SimpleBSONReader.increment_counter_dict(parsing_result.categories, category_id)
                SimpleBSONReader.increment_counter_dict(parsing_result.products, product_id)

                picture = Image.open(io.BytesIO(pic['picture']))

                SimpleBSONReader.picture_hash_count(
                    category_id,
                    pic,
                    parsing_result,
                    first_occurence_number=first_occurence_number)

                img_meta = ImgMeta(product_id, category_id, no_pic)

                img_pixel_matrix = Img2ArrayGray.transform(picture, save_png, img_meta, root_path)
                if pca_reduction:
                    # perform pca reduction
                    print("Performing pca reduction of the current img (%s, %s, %s)" % (product_id, category_id, no_pic))
                    img_pixel_matrix_reduced = Pipeline([
                        ('scaling', StandardScaler()),
                        ('pca', pca.PCA(n_components=n_components, ))]).fit_transform(img_pixel_matrix)
                    flatten_img_reduced = img_meta.__array__() + tuple(img_pixel_matrix_reduced.flatten())
                flatten_img = img_meta.__array__() + tuple(img_pixel_matrix.flatten())
                if parsing_result.pixel_matrix.shape == ():
                    parsing_result.pixel_matrix = np.array(flatten_img)
                    if pca_reduction:
                        parsing_result.pixel_matrix_reduced = np.array(flatten_img_reduced)
                else:
                    parsing_result.pixel_matrix = np.vstack((
                        parsing_result.pixel_matrix,
                        flatten_img
                    ))
                    if pca_reduction:
                        parsing_result.pixel_matrix_reduced = np.vstack((
                            parsing_result.pixel_matrix_reduced,
                            flatten_img_reduced
                        ))

            if save_intermediate:
                f = open(root_path + "csv/" + "full/%s-%s.csv" % (category_id, product_id), 'w')
                f.write(",".join(map(str, flatten_img)))
                if pca_reduction:
                    f_red = open(root_path + "csv/" + "reduced/pca_%s-%s.csv" % (category_id, product_id), 'w')
                    f_red.write(",".join(map(str, flatten_img_reduced)))

        if save_intermediate:
            with open(root_path + "csv/" + "total.csv", 'w') as f:
                f.write("%i" % total_pic_rows)
            SimpleBSONReader.write_dict_to_file("categories.csv", parsing_result.categories, root_path=root_path)
            SimpleBSONReader.write_dict_to_file("products.csv", parsing_result.products, root_path=root_path)
            SimpleBSONReader.write_dict_to_file("pictures.csv", parsing_result.pictures, root_path=root_path)
            SimpleBSONReader.write_dict_to_file("twin_pictures.csv", parsing_result.twin_pictures, root_path=root_path)

            SimpleBSONReader.write_df_to_file("pictures_categories.csv", parsing_result.pictures_categories, root_path=root_path)

            # clean picture_categories means keep only same (picture, category) couple
            pictures_categories_cleaned = parsing_result.pictures_categories[
                parsing_result.pictures_categories['count'] != first_occurence_number
            ]
            SimpleBSONReader.write_df_to_file("pictures_categories_cleaned.csv", pictures_categories_cleaned, root_path=root_path)
        return parsing_result

    @staticmethod
    def write_dict_to_file(file_name, array, root_path):
        with open(root_path + "csv/" + file_name, 'w') as file:
            SimpleBSONReader.write_csv_dictionary(file, array)

    @staticmethod
    def write_df_to_file(file_name, array, root_path):
        with open(root_path + "csv/" + file_name, 'w') as file:
            SimpleBSONReader.write_csv_df(file, array)

    @staticmethod
    def picture_hash_count(category_id, pic, parsing_result, first_occurence_number):
        md5 = hashlib.md5()
        md5.update(pic['picture'])
        md5_strip = md5.hexdigest().strip()
        category_id = str(category_id)
        index = category_id.__str__() + md5_strip

        if index in parsing_result.pictures_categories.index:
            parsing_result.pictures_categories.loc[index] = \
                [category_id, md5_strip, parsing_result.pictures_categories.loc[index]['count'] + 1]
        else:
            parsing_result.pictures_categories.loc[index] = [category_id, md5_strip, first_occurence_number]
            if sum(parsing_result.pictures_categories['pic_hash'] == md5_strip) > 1:
                SimpleBSONReader.increment_counter_dict(parsing_result.twin_pictures, md5_strip)

        SimpleBSONReader.increment_counter_dict(parsing_result.pictures, md5_strip)

    @staticmethod
    def add_pic_categ_row(index, parsing_result, pict_cat_row):
        parsing_result.pictures_categories.loc[index] = pict_cat_row

    @staticmethod
    def write_csv_dictionary(csv_file, dictionary):
        items = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True)
        csv.writer(csv_file).writerows(items)

    @staticmethod
    def write_csv_df(csv_file, df):
        df.to_csv(csv_file, index=False, header=False)
