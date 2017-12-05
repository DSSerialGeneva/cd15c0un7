from src.transformers.img.img_2_array_gray import Img2ArrayGray


class Img2ArrayGrayScaled:
    @staticmethod
    def transform(img):
        return Img2ArrayGray.transform(img) / 255


