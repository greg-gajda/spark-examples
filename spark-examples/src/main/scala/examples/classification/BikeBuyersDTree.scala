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
package examples.classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.tree.DecisionTree

import examples.common.Application.configLocalMode
import examples.PrintUtils.printMetrics
import examples.common.DataLoader.localFile
import examples.classification.Stats.confusionMatrix

object BikeBuyersDTree {

  def main(args: Array[String]): Unit = {

    val sc = new SparkContext(configLocalMode("Classification of Bike Buyers with DecisionTree"))
    val bbFile = localFile("bike-buyers.txt")(sc)

    val data = bbFile.map { row =>
      BikeBuyerModel(row.split("\\t")).toLabeledPoint
    }

    val Array(train, test) = data.randomSplit(Array(.9, .1), 102059L)
    train.cache()
    test.cache()

    val numClasses = 2
    val impurity = "entropy" 
    val maxDepth = 20
    val maxBins = 34

    val dtree = DecisionTree.trainClassifier(train, numClasses, BikeBuyerModel.categoricalFeaturesInfo(), impurity, maxDepth, maxBins)

    test.take(5).foreach {
      x => println(s"Predicted: ${dtree.predict(x.features)}, Label: ${x.label}")
    }

    val predictionsAndLabels = test.map {
      point => (dtree.predict(point.features), point.label)
    }

    val stats = Stats(confusionMatrix(predictionsAndLabels))
    println(stats.toString)
    
    val metrics = new BinaryClassificationMetrics(predictionsAndLabels)
    printMetrics(metrics)

    sc.stop()
  }

}