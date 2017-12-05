import sys
import pymongo_spark
from pyspark import SparkContext, SparkConf
from PIL import Image
from cStringIO import StringIO

# Print job arguments
print sys.argv

def jpeg_binary_to_grayscale_array( jpeg ):
  str_jpeg = StringIO(jpeg)
  image = Image.open(str_jpeg)
  grayscale_image = image.convert('L')
  grayscale_image_pixels = grayscale_image.getdata()
  return str(list(grayscale_image_pixels))

if __name__ == "__main__":

  # create Spark context with Spark configuration
  conf = SparkConf().setAppName("Spark Job")
  sc = SparkContext(conf=conf)

  # -------------------------------------
  # 1- output first item
  # pymongo_spark.activate()
  # bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
  # first_item = bsonFileRdd.take(1)
  # sc.parallelize(first_item).saveAsTextFile(sys.argv[2])

  # -------------------------------------
  # 2- print ID and Category ID of the first item
  # pymongo_spark.activate()
  # bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
  # first_item = bsonFileRdd.take(1)
  # rdd_first_item = sc.parallelize(first_item)
  # print "[DEBUG] ID=["+str(first_item[0]['_id'])+"], CATEGORY_ID=["+str(first_item[0]['category_id'])+"]"

  # -------------------------------------
  # 3- transform JPEG binary to RGB matrix and print to standard output
  # pymongo_spark.activate()
  # bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
  # first_item = bsonFileRdd.take(1)
  # rdd_first_item = sc.parallelize(first_item)
  # print "[DEBUG] ID=["+str(first_item[0]['_id'])+"], CATEGORY_ID=["+str(first_item[0]['category_id'])+"]"
  # binary_jpeg = first_item[0]['imgs'][0]['picture']
  # imgfile = StringIO(binary_jpeg)
  # img = Image.open(imgfile)
  # img = img.convert('L')
  # print str(list(img.getdata()))

  # -------------------------------------
  # 4- transform first image JPEG binary to RGB matrix and output
  # pymongo_spark.activate()
  # bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
  # first_item = bsonFileRdd.take(1)
  # rdd_first_item = sc.parallelize(first_item)
  # rdd_result = rdd_first_item.map(lambda x: ( str(x['_id'])+"_1", str(x['category_id']), str(list(Image.open(StringIO(x['imgs'][0]['picture'])).convert('L').getdata()))))
  # rdd_result.saveAsTextFile(sys.argv[2])

  # -------------------------------------
  # 5- transform all bson, only 1st image of each product
  # pymongo_spark.activate()
  # bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
  # rdd_result = bsonFileRdd.map(lambda x: ( str(x['_id'])+"_1", str(x['category_id']), str(list(Image.open(StringIO(x['imgs'][0]['picture'])).convert('L').getdata()))))
  # rdd_result.saveAsTextFile(sys.argv[2])

  # -------------------------------------
  # 6- transform all bson, all images of each product
  pymongo_spark.activate()
  bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
  rdd_result = bsonFileRdd.flatMap(lambda x: [(str(x['_id'])+"_"+str(ind), str(x['category_id']), jpeg_binary_to_grayscale_array(y['picture'])) for ind, y in enumerate(x['imgs'])])
  rdd_result.saveAsTextFile(sys.argv[2])