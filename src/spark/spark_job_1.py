import sys
from pyspark import SparkContext, SparkConf
import sparkdl
from sparkdl import readImages
from pyspark.sql.functions import lit, udf
import sys
import pymongo_spark
import io
from PIL import Image
from cStringIO import StringIO
import numpy as np
from pyspark import Row
from operator import add
from collections import namedtuple
from pyspark.sql.types import (BinaryType, IntegerType, StringType, StructField, StructType)
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

conf = SparkConf().setAppName("Spark Job")
sc = SparkContext(conf=conf)

#if __name__ == "__main__":

pymongo_spark.activate()
bsonFileRdd = sc.BSONFileRDD(sys.argv[1])
ImageType = namedtuple("ImageType", ["nChannels", "dtype", "channelContent", "pilMode", "sparkMode"])

class SparkMode(object):
    RGB = "RGB"
    FLOAT32 = "float32"
    RGB_FLOAT32 = "RGB-float32"

supportedImageTypes = [ImageType(3, "uint8", "RGB", "RGB", SparkMode.RGB), ImageType(1, "float32", "I", "F", SparkMode.FLOAT32), ImageType(3, "float32", "RGB", None, SparkMode.RGB_FLOAT32)]
pilModeLookup = {t.pilMode: t for t in supportedImageTypes if t.pilMode is not None}
sparkModeLookup = {t.sparkMode: t for t in supportedImageTypes}

def imageArrayToStruct(imgArray, sparkMode=None):
    if len(imgArray.shape) == 4:
        if imgArray.shape[0] != 1:
            raise ValueError("The first dimension of a 4-d image array is expected to be 1.")
        imgArray = imgArray.reshape(imgArray.shape[1:])
    if sparkMode is None:
        sparkMode = _arrayToSparkMode(imgArray)
    imageType = sparkModeLookup[sparkMode]
    height, width, nChannels = imgArray.shape
    if imageType.nChannels != nChannels:
        msg = "Image of type {} should have {} channels, but array has {} channels."
        raise ValueError(msg.format(sparkMode, imageType.nChannels, nChannels))
    if not np.can_cast(imgArray, imageType.dtype, 'same_kind'):
        msg = "Array of type {} cannot safely be cast to image type {}."
        raise ValueError(msg.format(imgArray.dtype, imageType.dtype))
    imgArray = np.array(imgArray, dtype=imageType.dtype, copy=False)
    data = bytearray(imgArray.tobytes())
    return Row(mode=sparkMode, height=height, width=width, nChannels=nChannels, data=data)

def _decodeImage(imageData):
    try:
        img = Image.open(io.BytesIO(imageData))
    except IOError:
        return None
    if img.mode in pilModeLookup:
        mode = pilModeLookup[img.mode]
    else:
        msg = "We don't currently support images with mode: {mode}"
        warn(msg.format(mode=img.mode))
        return None
    imgArray = np.asarray(img)
    image = imageArrayToStruct(imgArray, mode.sparkMode)
    return image

from pyspark.sql.types import (BinaryType, IntegerType, StringType, StructField, StructType)
imageSchema = StructType([StructField("mode", StringType(), False), StructField("height", IntegerType(), False), StructField("width", IntegerType(), False), StructField("nChannels", IntegerType(), False), StructField("data", BinaryType(), False)])
decodeImage = udf(_decodeImage, imageSchema)
rdd_result = bsonFileRdd.flatMap(lambda x: [(str(x['_id'])+"_"+str(ind)+"_"+str(len(y['picture'])), int(x['category_id']), bytearray(y['picture'])) for ind, y in enumerate(x['imgs'])])
rdd_saved = rdd_result
rdd_result_temp = sc.parallelize(rdd_result.map(lambda (x,y,z): (y, 1)).reduceByKey(add).sortBy(lambda x: x[1], ascending=False).take(2)).map(lambda (x,y): x)
categories = list(rdd_result_temp.distinct().collect())
dict_categories = {k: v for v, k in enumerate(categories)}

def get_index_cat(x):
    return dict_categories[x]

rdd_result_filtered = rdd_saved.filter(lambda (x,y,z): y in categories)
spark = SparkSession(sc)
dataframe_result0 = rdd_result_filtered.map(lambda (x,y,z): (x, get_index_cat(y), z)).toDF()
dataframe_result1 = dataframe_result0.toDF("productID_imageID", "label", "image")
all_df = dataframe_result1.select("productID_imageID", decodeImage("image").alias("image"), "label")
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from sparkdl import DeepImageFeaturizer
featurizer = DeepImageFeaturizer(inputCol="image", outputCol="features", modelName="InceptionV3")
lr = LogisticRegression(maxIter=20, regParam=0.05, elasticNetParam=0.3, labelCol="label")
p = Pipeline(stages=[featurizer, lr])
all_df.show()
all_0_df = all_df.filter(all_df.label == 0)
all_1_df = all_df.filter(all_df.label == 1)
train_0_df, test_0_df = all_0_df.randomSplit([0.8, 0.2])
train_1_df, test_1_df = all_1_df.randomSplit([0.8, 0.2])
train_df = train_0_df.unionAll(train_1_df)
test_df = test_0_df.unionAll(test_1_df)
p_model = p.fit(train_df)

predictions = p_model.transform(test_df)
predictions.printSchema()
predictions.select("productID_imageID", "prediction", "rawPrediction").show(truncate=False)

predictionAndLabels = predictions.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

#l = sc.parallelize([1,2,3])
#l.saveAsTextFile(sys.argv[2])
