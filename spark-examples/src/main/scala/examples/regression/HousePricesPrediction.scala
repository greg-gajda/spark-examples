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
package examples.regression

import scala.util.control.Exception.catching
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import examples.common.Application.regressionApp
import examples.common.Application._
import examples.common.DataLoader._
import examples.PrintUtils.printMetrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import examples.classification.BikeBuyerModel
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.RegressionMetrics

object HousePricesPrediction {

  def printColumnsStats(rdd: RDD[LabeledPoint]) = {
    val stats = Statistics.colStats(rdd.map { x => x.features })
    println(s"Max : ${stats.max}")
    println(s"Min : ${stats.min}")
    println(s"Mean : ${stats.mean}")
    println(s"Variance : ${stats.variance}")    
  }
  
  def createLinearRegressionModel(rdd: RDD[LabeledPoint], numIterations: Int = 100, stepSize: Double = 0.01) = {
    LinearRegressionWithSGD.train(rdd, numIterations, stepSize)
  }

  def createDecisionTreeRegressionModel(rdd: RDD[LabeledPoint], maxDepth: Int = 10, maxBins: Int = 20) = {
    val impurity = "variance"
    DecisionTree.trainRegressor(rdd, Map[Int, Int](), impurity, maxDepth, maxBins)
  }

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(configYarnClientMode(regressionApp))
    val hdFile = hdfsFile("house-data.csv")(sc)

    val houses = hdFile.map(_.split(",")).
      filter { t => catching(classOf[NumberFormatException]).opt(t(0).toLong).isDefined }.
      map { HouseModel(_).toLabeledPoint() }

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(houses.map(dp => dp.features))

    val Array(train, test) = houses.
      map(dp => new LabeledPoint(dp.label, scaler.transform(dp.features))).
      randomSplit(Array(.9, .1), 10204L)

    val model = createDecisionTreeRegressionModel(train)

    test.take(5).foreach {
      x => println(s"Predicted: ${model.predict(x.features)}, Label: ${x.label}")
    }

    val predictionsAndValues = test.map {
      point => (model.predict(point.features), point.label)
    }

    println("Mean house price: " + test.map { x => x.label }.mean())
    println("Max prediction error: " + predictionsAndValues.map { case (p, v) => math.abs(v - p) }.max)

    val metrics = new RegressionMetrics(predictionsAndValues)

    println(s"Mean Squared Error: ${metrics.meanSquaredError}")
    println(s"Root Mean Squared Error: ${metrics.rootMeanSquaredError}")
    println(s"Coefficient of Determination R-squared: ${metrics.r2}")
    println(s"Mean Absoloute Error: ${metrics.meanAbsoluteError}")
    println(s"Explained variance: ${metrics.explainedVariance}")

    sc.stop
  }
}