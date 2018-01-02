from PIL import ImageDraw

from src.transformers.img.img_2_array import Img2Array


class Img2ArrayGray:
    @staticmethod
    def transform(img, save=False, img_meta=''):
        img_gray = img.convert('L')
        if save:
            ImageDraw.Draw(img_gray)
            img_gray.save("../../out/gray/img-" + img_meta.__str__() + ".png")
        return Img2Array.transform(img_gray)


