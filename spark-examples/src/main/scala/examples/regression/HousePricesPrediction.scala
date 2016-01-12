package examples.regression

import scala.util.control.Exception.catching
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import examples.common.Application.regressionApp
import examples.common.Application.configLocalMode
import examples.common.DataLoader.localFile
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

  def createLinearRegressionModel(rdd: RDD[LabeledPoint], numIterations: Int = 100, stepSize: Double = 0.01) = {
    LinearRegressionWithSGD.train(rdd, numIterations, stepSize)
  }

  def createDecisionTreeRegressionModel(rdd: RDD[LabeledPoint], maxDepth: Int = 10, maxBins: Int = 20) = {
    val impurity = "variance"
    DecisionTree.trainRegressor(rdd, Map[Int, Int](), impurity, maxDepth, maxBins)
  }

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(configLocalMode(regressionApp))
    val hdFile = localFile("house-data.csv")(sc)

    val houses = hdFile.map(_.split(",")).
      filter { t => catching(classOf[NumberFormatException]).opt(t(0).toLong).isDefined }.
      map { HouseModel(_).toLabeledPoint() }

    val scaler = new StandardScaler(withMean = true, withStd = true).fit(houses.map(dp => dp.features))

    val Array(train, test) = houses.
      map(dp => new LabeledPoint(dp.label, scaler.transform(dp.features))).
      randomSplit(Array(.9, .1), 10204L)

    val stats2 = Statistics.colStats(train.map { x => x.features })
    println(s"Max : ${stats2.max}, Min : ${stats2.min}, and Mean : ${stats2.mean} and Variance : ${stats2.variance}")

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