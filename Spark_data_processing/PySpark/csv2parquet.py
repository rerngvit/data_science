from pyspark import SparkConf, SparkContext
from operator import add
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
import sys

'''
Spark job for converting a CSV file to a Parquet file.
It is designed for spark-submit script.
Usage:
  spark-submit csv2parquet.py  [HDFS path to input CSV file] \
              [HDFS path for output Parquet file]
'''


def main(sc, src_data_csv_path, output_parquet_path):
    def load_csv_to_df(file_path, gzip=True):
        ## Read Data using SQL Context
        if not gzip:
            df = sqlContext.read.format('com.databricks.spark.csv').options(
                header='true', inferschema='true').load(file_path)
        else:
            df = sqlContext.read.format('com.databricks.spark.csv').options(
                header='true', inferschema='true',
                codec="org.apache.hadoop.io.compress.GzipCodec").load(
                file_path)
        return df

    sqlContext = SQLContext(sc)
    data_df = load_csv_to_df(src_data_csv_path, gzip=True)
    data_df.write.parquet(output_parquet_path, mode="overwrite")
    print(" Conversion finish ")

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName("CSVToParquet")
    sc = SparkContext(conf=conf)

    # Setup file path
    src_data_csv_path = sys.argv[1]
    output_parquet_path = sys.argv[2]

    # Execute Main functionality
    main(sc, src_data_csv_path, output_parquet_path)
