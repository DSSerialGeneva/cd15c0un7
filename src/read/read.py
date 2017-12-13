import numpy

from src.readers import SimpleBSONReader

pixels = SimpleBSONReader.read_all('../../data/train_example.bson')

numpy.savetxt("foo.csv", pixels, delimiter=",", fmt='%.d')


# img_gray = img.convert('L')
#
# pixels = Img2Array.transform(img)
#
# pixels_gray = Img2ArrayGrayScaled.transform(img).flatten()
#
# draw = ImageDraw.Draw(img)
# img.save("../../out/a_test.png")
#
# draw_gray = ImageDraw.Draw(img_gray)
# img_gray.save("../../out/a_test_gray.png")

