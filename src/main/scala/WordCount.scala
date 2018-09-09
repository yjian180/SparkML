/**
  * Created by think on 2018/4/17.
  */
import org.apache.spark._
object WordCount {

  def main(args: Array[String]) {
    var masterUrl = "local[2]"
    var inputPath = "D:\\Program-code\\Sparktest\\data\\hamlet.txt"
    var outputPath = "D:\\Program-code\\Sparktest\\output"

    if (args.length == 1) {
      masterUrl = args(0)
    } else if (args.length == 3) {
      masterUrl = args(0)
      inputPath = args(1)
      outputPath = args(2)
    }

    println(s"masterUrl:${masterUrl}, inputPath: ${inputPath}, outputPath: ${outputPath}")

    val sparkConf = new SparkConf().setMaster(masterUrl).setAppName("WordCount")
    val sc = new SparkContext(sparkConf)

    val rowRdd = sc.textFile(inputPath)
    val resultRdd = rowRdd.flatMap(line => line.split("\\s+"))
      .map(word => (word, 1)).reduceByKey(_ + _)

    resultRdd.saveAsTextFile(outputPath)
  }
}
