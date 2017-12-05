from src.transformers.img.img_2_array import Img2Array


class Img2ArrayGray:
    @staticmethod
    def transform(img):
        img_gray = img.convert('L')
        return Img2Array.transform(img_gray)


