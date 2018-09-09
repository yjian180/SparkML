package mymllib

import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.rdd.RDD

/**
  * Created by zy on 16-5-20.
  */
class BinaryClassEvaluation(probabilityAndLabels: RDD[(Double, Double)]) {

  val binaryMetrics = new BinaryClassificationMetrics(probabilityAndLabels)
  val auc = binaryMetrics.areaUnderROC
  val aupr = binaryMetrics.areaUnderPR
  println(auc)
//  println("Area Under ROC:" + auc)
//  println("Area Under PR:" + aupr)
}
