# CSVToPIO
A Spark application for converting a CSV file to PredictionIO JSON format.

The application expects an input file, an output path, and the configuration file to be in HDFS.

How to compile and execute the application
*  Adjust your csv_to_pio.conf to your need and upload the configuration to HDFS
*  Compile by using Maven
   * $ mvn -f pom_uber.xml clean package -DskipTests
*  Run the application and specify the HDFS path of the config file as an argument
   * For example
   * $ SPARK_HOME/bin/spark-submit --class main.scala.CSVToPIO \
     --master {Spark Master URL} \
     --conf spark.app.id=CSVToPIO \
     ./target/CSVToPIO-1.0-SNAPSHOT.jar \
     hdfs://172.31.212.82:9000/conf/csv_to_pio.thin.conf
