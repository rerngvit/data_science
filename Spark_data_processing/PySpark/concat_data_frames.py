from pyspark import SparkConf, SparkContext
import pyspark
from operator import add
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from pyspark.sql.functions import *
import sys

'''
Spark job for concatanating two dataframes by rows.
The dataframes do not need to have exact same columns.
It is designed for spark-submit script.
Usage:
  spark-submit concat_data_frames.py  [HDFS path to Parquet 1] \
              [HDFS path to Parquet 2] [HDFS path output]
The outcome will contain rows from both dataframes and the
intersection of columns between the two dataframes.
'''


def main(sc, src_parquet_path, dest_parquet_path, joined_parquet_path):
    sqlContext = SQLContext(sc)
    # Read in source and destination dataframes
    src_df = sqlContext.read.parquet(src_parquet_path)
    dest_df = sqlContext.read.parquet(dest_parquet_path)

    # Merge the columns between the dataframes
    target_columns = sorted(list(set(src_df.columns)
                                 .intersection(set(dest_df.columns))))
    joined_df = src_df.select(target_columns).unionAll(
        dest_df.select(target_columns))
    joined_df.write.parquet(joined_parquet_path, mode="overwrite")

if __name__ == "__main__":
    # Configure Spark
    conf = SparkConf().setAppName("Concat_data_frames")
    sc = SparkContext(conf=conf)

    # Setup file path
    src_parquet_path = sys.argv[1]
    dest_parquet_path = sys.argv[2]
    joined_parquet_path = sys.argv[3]

    # Execute Main functionality
    main(sc, src_parquet_path, dest_parquet_path, joined_parquet_path)
