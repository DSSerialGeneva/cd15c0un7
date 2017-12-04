import io

import numpy as np
from PIL import Image
from PIL import ImageDraw

from src.readers import SimpleBSONReader

reader = SimpleBSONReader('../../data/train_example.bson')

imageBytes = reader.read()['imgs'][0]['picture']


stream = io.BytesIO(imageBytes)

img = Image.open(stream)
img_gray = img.convert('L')

pixels = np.array(img)

pixels_gray = np.array(img_gray) / 255

draw = ImageDraw.Draw(img)
img.save("../../out/a_test.png")

draw_gray = ImageDraw.Draw(img_gray)
img_gray.save("../../out/a_test_gray.png")

