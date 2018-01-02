import io

import bson
import numpy as np

from PIL import Image

from src.domain.img_meta import ImgMeta

from src.transformers.img.img_2_array_gray import Img2ArrayGray


class SimpleBSONReader:

    @staticmethod
    def read_first(file_name):
        bson_file = open(file_name, 'rb')

        image_bytes = bson.loads(bson_file.read())['imgs'][0]['picture']

        stream = io.BytesIO(image_bytes)

        return Image.open(stream)

    @staticmethod
    def read_all(file_name):
        bson_file = open(file_name, 'rb')

        bson_iterator = bson.decode_file_iter(bson_file)

        pixel_matrix = np.empty(())
        for c, d in enumerate(bson_iterator):
            product_id = d['_id']
            category_id = d['category_id']
            for no_pic, pic in enumerate(d['imgs']):
                picture = Image.open(io.BytesIO(pic['picture']))

                img_meta = ImgMeta(product_id, category_id, no_pic)

                if pixel_matrix.shape == ():
                    pixel_matrix = np.array(img_meta.__array__() + tuple(Img2ArrayGray.transform(picture, True, img_meta).flatten()))
                else:
                    pixel_matrix = np.vstack((
                        pixel_matrix,
                        img_meta.__array__() + tuple(Img2ArrayGray.transform(picture, True, img_meta).flatten())
                    ))
        return pixel_matrix

