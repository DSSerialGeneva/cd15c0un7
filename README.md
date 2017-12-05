# cd15c0un7

## Steps to install Spark locally on Linux

1. Install Python 2
2. Install Java 8
3. Install Scala
4. Install Spark Hadoop
5. Install required libraries to process MongoDB BSON

## To run a Spark job locally

```
spark-submit --jars mongo-hadoop-spark-2.0.2.jar job_transform_bson_to_grayscale_matrix.py train_example.bson output
```