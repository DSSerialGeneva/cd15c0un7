from PIL import ImageDraw

from src.transformers.img.img_2_array import Img2Array


class Img2ArrayGray:
    @staticmethod
    def transform(img, save=False, img_meta='', root_path="../../out/"):
        img_gray = img.convert('L')
        if save:
            ImageDraw.Draw(img_gray)
            img_gray.save("%s/gray/img-%s.png" % (root_path, img_meta.__str__()))
        return Img2Array.transform(img_gray)


