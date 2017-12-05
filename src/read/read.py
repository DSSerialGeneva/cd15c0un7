from PIL import ImageDraw
from src.readers import SimpleBSONReader
from src.transformers.img.img_2_array import Img2Array
from src.transformers.img.img_2_array_gray_scaled import Img2ArrayGrayScaled

img = SimpleBSONReader.read('../../data/train_example.bson')
img_gray = img.convert('L')

pixels = Img2Array.transform(img)

pixels_gray = Img2ArrayGrayScaled.transform(img).flatten()

draw = ImageDraw.Draw(img)
img.save("../../out/a_test.png")

draw_gray = ImageDraw.Draw(img_gray)
img_gray.save("../../out/a_test_gray.png")

