import bson
import io
from PIL import Image


class SimpleBSONReader:

    @staticmethod
    def read(file_name):
        bson_file = open(file_name, 'rb')

        image_bytes = bson.loads(bson_file.read())['imgs'][0]['picture']

        stream = io.BytesIO(image_bytes)

        return Image.open(stream)
