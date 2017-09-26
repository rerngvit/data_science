package main.scala

import java.io.{BufferedReader, InputStreamReader}
import com.typesafe.config.{Config, ConfigFactory}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FSDataInputStream, FileSystem}
import org.apache.spark.{SparkConf, SparkContext}
import scala.collection.mutable.ArrayBuffer

object CSVToPIO {
  var sc: SparkContext = null
  var appConf: Config = null
  var hdfs: FileSystem = null

  def init(configPath: String): Unit = {
    def initHDFS(): Unit = {
      val conf = new Configuration()
      conf.set("fs.defaultFS", configPath)
      hdfs = FileSystem.get(conf)
    }

    def initAppConfig(): Unit = {
      val fis: FSDataInputStream = hdfs.open(new org.apache.hadoop.fs.Path(configPath))
      val reader = new BufferedReader(new InputStreamReader(fis))
      appConf = ConfigFactory.parseReader(reader)
    }

    def initSparkContext(): Unit = {
      val conf = new SparkConf().setAppName(appConf.getString("configuration.Spark.ApplicationName"))
      sc = new SparkContext(conf)
   }

    initHDFS()
    initAppConfig()
    initSparkContext()
  }

  def cleanHeaderItem(rawCol: String): String = {
    rawCol.replaceAll("^\"", "").replaceAll("\"$", "")
  }


  def main(args: Array[String]): Unit = {
    val configPath = args(0)
    init(configPath)
    val src = sc.textFile(appConf.getString("configuration.IO.inputPath"))
    val columnNames = src.first().split(",").map(cleanHeaderItem)
    def formatStringPair(key: String, value: String): String = {
      def addQuotes(s: String): String = {
        "\"" + s + "\""
      }
      addQuotes(key) + " : " + addQuotes(value)
    }
    val event = appConf.getString("configuration.PredictionIO.event")
    val entityType = appConf.getString("configuration.PredictionIO.entityType")
    val entityId = appConf.getString("configuration.PredictionIO.entityId")
    val out = src.zipWithIndex().filter(_._2 > 0).map(_._1).map { raw =>
      val row = raw.split(",")
      val rowOut = ArrayBuffer[String]()
      rowOut.append(formatStringPair("event", event))
      rowOut.append(formatStringPair("entityType", entityType))
      rowOut.append(formatStringPair("entityId", entityId))

      val propertiesOut = new Array[String](row.length)
      for (j <- 0 until row.length) {
        propertiesOut(j) = formatStringPair(columnNames(j), row(j))
      }
      val propertiesJSONString = "\"properties\"" + ":" + propertiesOut.mkString("{", ", ", "}")
      rowOut.append(propertiesJSONString)
      rowOut.mkString("{", ", ", "}")
    }
    val outputPath = appConf.getString("configuration.IO.outputPath")
    val fsPath = new org.apache.hadoop.fs.Path(outputPath)
    if (hdfs.exists(fsPath)) hdfs.delete(fsPath, true)
    out.coalesce(appConf.getInt("configuration.IO.numPartitions")).saveAsTextFile(outputPath)
  }
}