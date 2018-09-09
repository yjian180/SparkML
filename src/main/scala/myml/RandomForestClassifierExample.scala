/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println
package myml

// $example on$
import mymllib.MultiClassEvaluation
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Row
// $example off$
import org.apache.spark.sql.SparkSession

object RandomForestClassifierExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder
      .appName("RandomForestClassifierExample")
      .master("local[2]")
      .getOrCreate()
import spark.implicits._
    // $example on$
    // Load and parse the data file, converting it to a DataFrame.
    val dataname = "cleveland"
//    val data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
    val data = spark.read.format("libsvm").load("data/" + dataname + ".libsvmForm")

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val datasetClass = data.select("label").collect().distinct
    val classnum = datasetClass.size
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
//      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    val kfold = 10
    val paramGrid = new ParamGridBuilder()
      .addGrid(featureIndexer.maxCategories, Array(classnum))
      .addGrid(rf.numTrees, Array(1, 3, 5, 10, 20))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(kfold)

    val cvModel = cv.fit(trainingData)

    val rfModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(2).asInstanceOf[RandomForestClassificationModel]
    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")

    // Train model. This also runs the indexers.
    val predictions = cvModel.transform(testData)
//    val model = pipeline.fit(trainingData)
//    val mypredictionAndLables = predictions.select("prediction", "label").map(row =>
//     (row.getAs[Double]("prediction"), row.getAs[Double]("label")))


    // Make predictions.
//    val predictions = Model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "label", "features").show(100)

//    new MulticlassClassificationEvaluatorSuite()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//      .setMetricName("f1")
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//      .setMetricName("weightedPrecision")
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//      .setMetricName("weightedRecall")

//    val f1 = evaluator.evaluate(predictions)
//    println(s"f1 = ${f1}")
//    val weightedRecall = evaluator.evaluate(predictions)
//    println(s"recall = ${weightedRecall}")
//    val weightedPrecision = evaluator.evaluate(predictions)
//    println(s"Precision = ${weightedPrecision}")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${1-accuracy}")

    // Select (prediction, true label) and compute test error.
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//      .setMetricName("f1")
//      .setMetricName("weightedPrecision")
//      .setMetricName("weightedRecall")
//
//    val f1 = evaluator.evaluate(predictions)
//      println(s"f1 = ${f1}")
//    val weightedRecall = evaluator.evaluate(predictions)
//      println(s"recall = ${weightedRecall}")
//    val weightedPrecision = evaluator.evaluate(predictions)
//      println(s"Precision = ${weightedPrecision}")
//    val accuracy = evaluator.evaluate(predictions)
//      println(s"Test Error = ${accuracy}")
//


    //output parameters of the best model
    cvModel.bestModel.params.foreach(println)
    val paramMap = cvModel.getEstimatorParamMaps
      .zip(cvModel.avgMetrics)
      .maxBy(_._2)
      ._1
    println(paramMap)


//    val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
//    println(s"Learned classification forest model:\n ${rfModel.toDebugString}")
    // $example off$

    spark.stop()
  }
}
// scalastyle:on println
